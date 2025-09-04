#!/usr/bin/env python3
"""
BERT Token Classification Inference Service for HuggingFace Space.

CPU-optimized BERT service for PII detection with HuggingFace Hub support.
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings("ignore", message=".*FutureWarning.*")

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from pii_masking.text_processing import PIIPrediction, EntitySpan, reconstruct_masked_text

from .base_service import BasePIIInferenceService

logger = logging.getLogger(__name__)

class BERTInferenceService(BasePIIInferenceService):
    """Production-ready BERT Token Classification Service for PII detection."""
    
    def __init__(self, model_path: str, config: Dict = None):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 4)
        
        logger.info(f"BERT service initialized with model path: {model_path}")
        
    async def initialize(self) -> bool:
        """Initialize the model and tokenizer for CPU inference."""
        try:
            logger.info("Loading BERT model and tokenizer...")
            
            logger.info(f"Downloading model from HuggingFace Hub: {self.model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            torch.set_num_threads(2)
            
            self.is_initialized = True
            
            logger.info("BERT model loaded successfully")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            return False
    
    def predict_sync(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """Synchronous prediction method for BERT (CPU-bound)."""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
        
        try:
            start_time = time.time()
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1).squeeze().tolist()
            
            if isinstance(predicted_token_class_ids, int):
                predicted_token_class_ids = [predicted_token_class_ids]
            
            entities_dict, spans = self._convert_predictions_to_entities(
                text, inputs, predicted_token_class_ids
            )
            
            masked_text = reconstruct_masked_text(text, entities_dict)
            
            prediction = PIIPrediction(
                entities=entities_dict,
                spans=spans,
                masked_text=masked_text,
                original_text=text
            )
            
            inference_time = time.time() - start_time
            logger.debug(f"BERT inference time: {inference_time:.3f}s")
            
            if pii_entities is not None:
                prediction = self._filter_prediction_by_entities(prediction, pii_entities)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during BERT prediction: {e}")
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
    
    def _convert_predictions_to_entities(self, text: str, inputs: Dict, predicted_ids: List[int]) -> tuple:
        """Convert token-level predictions to entity dictionary and spans."""
        
        offset_mapping = inputs['offset_mapping'].squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        
        if not isinstance(offset_mapping[0], list):
            offset_mapping = [offset_mapping]
        if isinstance(tokens, str):
            tokens = [tokens]
        
        raw_spans = []
        
        for i, (token, pred_id, (start, end)) in enumerate(zip(tokens, predicted_ids, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]'] or start == end == 0:
                continue
                
            pred_label = self.model.config.id2label[pred_id]
            
            if pred_label != 'O':
                raw_spans.append({
                    'entity_type': pred_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end].strip()
                })
        
        merged_spans = self._merge_adjacent_spans(text, raw_spans)
        
        entities_dict = {}
        final_spans = []
        
        for span_data in merged_spans:
            entity_type = span_data['entity_type']
            entity_text = span_data['text']
            
            if entity_type not in entities_dict:
                entities_dict[entity_type] = []
            entities_dict[entity_type].append(entity_text)
            
            final_spans.append(EntitySpan(
                entity_type=entity_type,
                start=span_data['start'],
                end=span_data['end'],
                text=entity_text
            ))
        
        return entities_dict, final_spans
    
    def _merge_adjacent_spans(self, text: str, raw_spans: List[Dict]) -> List[Dict]:
        """Merge adjacent spans of the same entity type."""
        if not raw_spans:
            return []
        
        raw_spans.sort(key=lambda x: x['start'])
        merged = []
        current_span = raw_spans[0].copy()
        
        for next_span in raw_spans[1:]:
            if (next_span['entity_type'] == current_span['entity_type'] and 
                next_span['start'] <= current_span['end'] + 2):
                
                current_span['end'] = max(current_span['end'], next_span['end'])
                current_span['text'] = text[current_span['start']:current_span['end']].strip()
            else:
                merged.append(current_span)
                current_span = next_span.copy()
        
        merged.append(current_span)
        return merged
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for health checks."""
        return {
            "service_type": "bert_token_classification",
            "model_path": self.model_path,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "initialized": self.is_initialized,
            "num_parameters": self.model.num_parameters() if self.model else None,
            "num_labels": self.model.config.num_labels if self.model else None
        }

async def create_bert_service(model_path: str, config: Dict = None) -> BERTInferenceService:
    """Factory function to create and initialize BERT service."""
    service = BERTInferenceService(model_path, config)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize BERT service")
    
    return service

async def test_service():
    """Test function for development."""
    try:
        service = await create_bert_service("SoelMgd/bert-pii-detection")
        
        test_text = "Hi, my name is John Smith and my email is john.smith@company.com"
        result = await service.predict(test_text)
        
        print(f"Original: {test_text}")
        print(f"Masked: {result.masked_text}")
        print(f"Entities: {result.entities}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_service()) 