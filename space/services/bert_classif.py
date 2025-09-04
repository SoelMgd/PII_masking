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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
)

# Import our existing module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from pii_masking.text_processing import PIIPrediction, EntitySpan, reconstruct_masked_text

# Import base service
from .base_service import BasePIIInferenceService

# Setup logging
logger = logging.getLogger(__name__)

class BERTInferenceService(BasePIIInferenceService):
    """
    Production-ready BERT Token Classification Service for PII detection.
    
    Optimized for CPU inference in HuggingFace Spaces with HF Hub support.
    """
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Initialize the BERT inference service.
        
        Args:
            model_path: Path to the fine-tuned BERT model (local path or HF Hub repository)
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")  # Force CPU for HF Spaces
        
        # Performance settings optimized for CPU
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 4)  # Smaller batch for CPU
        
        logger.info(f"🔧 BERT service initialized with model path: {model_path}")
        
    async def initialize(self) -> bool:
        """Initialize the model and tokenizer for CPU inference."""
        try:
            logger.info("🚀 Loading BERT model and tokenizer...")
            
            # Load model and tokenizer from HuggingFace Hub
            logger.info(f"📥 Downloading model from HuggingFace Hub: {self.model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=False  # Security: don't execute remote code
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False
            )
            
            # Move to CPU and optimize for inference
            self.model.to(self.device)
            self.model.eval()
            
            # CPU optimization
            torch.set_num_threads(2)  # Use 2 threads for better performance
            
            self.is_initialized = True
            
            logger.info(f"✅ BERT model loaded successfully")
            logger.info(f"📊 Model parameters: {self.model.num_parameters():,}")
            logger.info(f"🏷️  Number of labels: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize BERT model: {e}")
            return False
    
    def predict_sync(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """
        Synchronous prediction method for BERT (CPU-bound).
        This is the natural way to call PyTorch/Transformers models.
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
        
        try:
            start_time = time.time()
            
            # Tokenize input with offset mapping
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )
            
            # Move to device (CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1).squeeze().tolist()
            
            # Handle single token case
            if isinstance(predicted_token_class_ids, int):
                predicted_token_class_ids = [predicted_token_class_ids]
            
            # Convert token predictions to PII entities and spans
            entities_dict, spans = self._convert_predictions_to_entities(
                text, inputs, predicted_token_class_ids
            )
            
            # Reconstruct masked text using the existing function
            masked_text = reconstruct_masked_text(text, entities_dict)
            
            # Create PIIPrediction object
            prediction = PIIPrediction(
                entities=entities_dict,
                spans=spans,
                masked_text=masked_text,
                original_text=text
            )
            
            inference_time = time.time() - start_time
            logger.debug(f"⚡ BERT inference time: {inference_time:.3f}s")
            
            # Filter entities if requested
            if pii_entities is not None:
                prediction = self._filter_prediction_by_entities(prediction, pii_entities)
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error during BERT prediction: {e}")
            # Return empty prediction on error
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
    
    def _convert_predictions_to_entities(self, text: str, inputs: Dict, predicted_ids: List[int]) -> tuple:
        """Convert token-level predictions to entity dictionary and spans."""
        
        offset_mapping = inputs['offset_mapping'].squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        
        # Handle single token case
        if not isinstance(offset_mapping[0], list):
            offset_mapping = [offset_mapping]
        if isinstance(tokens, str):
            tokens = [tokens]
        
        # First pass: collect all individual token predictions
        raw_spans = []
        
        for i, (token, pred_id, (start, end)) in enumerate(zip(tokens, predicted_ids, offset_mapping)):
            # Skip special tokens and empty spans
            if token in ['[CLS]', '[SEP]', '[PAD]'] or start == end == 0:
                continue
                
            pred_label = self.model.config.id2label[pred_id]
            
            if pred_label != 'O':  # PII token
                raw_spans.append({
                    'entity_type': pred_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end].strip()
                })
        
        # Second pass: merge adjacent spans of the same type
        merged_spans = self._merge_adjacent_spans(text, raw_spans)
        
        # Third pass: create entities dict and final spans
        entities_dict = {}
        final_spans = []
        
        for span_info in merged_spans:
            entity_type = span_info['entity_type']
            entity_text = span_info['text']
            
            # Add to entities dictionary
            if entity_type not in entities_dict:
                entities_dict[entity_type] = []
            if entity_text not in entities_dict[entity_type]:
                entities_dict[entity_type].append(entity_text)
            
            # Add to spans list
            final_spans.append(EntitySpan(
                entity_type=entity_type,
                start=span_info['start'],
                end=span_info['end'],
                text=entity_text
            ))
        
        return entities_dict, final_spans
    
    def _merge_adjacent_spans(self, text: str, raw_spans: List[Dict]) -> List[Dict]:
        """
        Merge adjacent spans of the same entity type, including those separated by whitespace.
        """
        if not raw_spans:
            return []
        
        # Sort spans by start position
        sorted_spans = sorted(raw_spans, key=lambda x: x['start'])
        merged_spans = []
        
        current_span = sorted_spans[0].copy()
        
        for next_span in sorted_spans[1:]:
            # Check if spans are of the same type and should be merged
            if (current_span['entity_type'] == next_span['entity_type'] and 
                self._should_merge_spans(text, current_span, next_span)):
                
                # Merge spans: extend the current span to include the next one
                current_span['end'] = next_span['end']
                current_span['text'] = text[current_span['start']:current_span['end']].strip()
                
            else:
                # Different type or too far apart, save current and start new
                merged_spans.append(current_span)
                current_span = next_span.copy()
        
        # Don't forget the last span
        merged_spans.append(current_span)
        
        return merged_spans
    
    def _should_merge_spans(self, text: str, span1: Dict, span2: Dict) -> bool:
        """
        Determine if two spans should be merged.
        """
        if span1['entity_type'] != span2['entity_type']:
            return False
        
        # Get the text between the spans
        between_start = span1['end']
        between_end = span2['start']
        
        if between_end <= between_start:
            return True  # Adjacent or overlapping
        
        between_text = text[between_start:between_end]
        entity_type = span1['entity_type']
        
        # More aggressive merging for specific entity types
        if entity_type in ['PHONENUMBER', 'SSN', 'ACCOUNTNAME']:
            # For phone numbers, SSNs, and account numbers, merge if separated by:
            # - No gap (adjacent tokens)
            # - Common phone/ID separators: spaces, dashes, parentheses, dots
            if len(between_text) <= 5 and all(c in ' \t\n()-.' for c in between_text):
                return True
        
        elif entity_type in ['FIRSTNAME', 'LASTNAME', 'MIDDLENAME']:
            # For names, be more conservative - only merge with single space or initials
            if len(between_text) <= 2 and between_text.strip() in ['', '.']:
                return True
        
        elif entity_type in ['STREET', 'SECONDARYADDRESS', 'CITY']:
            # For addresses, merge with spaces and common separators
            if len(between_text) <= 3 and all(c in ' \t\n,-' for c in between_text):
                return True
        
        elif entity_type in ['DATE', 'TIME']:
            # For dates and times, merge with spaces, commas, and common separators
            if len(between_text) <= 4 and all(c in ' \t\n,/-:' for c in between_text):
                return True
        
        elif entity_type in ['EMAIL', 'URL']:
            # For emails and URLs, only merge if directly adjacent (no gaps allowed)
            if len(between_text) == 0:
                return True
        
        else:
            # Default behavior: merge if only whitespace and simple punctuation
            if len(between_text) <= 3 and between_text.strip() in ['', ',', '.', '-', '/', ':', ';']:
                return True
            
            # Also merge if it's just whitespace
            if between_text.isspace() and len(between_text) <= 2:
                return True
        
        return False
    

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for monitoring."""
        info = {
            "service_name": "BERTInferenceService",
            "model_path": self.model_path,
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "config": self.config,
            "description": "BERT-based PII token classification with CPU optimization"
        }
        
        if self.is_initialized and self.model:
            info.update({
                "num_parameters": self.model.num_parameters(),
                "num_labels": self.model.config.num_labels,
                "max_length": self.max_length
            })
        
        return info

# Factory function for easy initialization
async def create_bert_service(model_path: str, config: Dict = None) -> BERTInferenceService:
    """
    Factory function to create and initialize BERT service.
    
    Args:
        model_path: Path to the fine-tuned BERT model (local path or HuggingFace Hub repository)
        config: Optional configuration dictionary
        
    Returns:
        Initialized BERTInferenceService
    """
    # Only check for local files if it's not a HuggingFace Hub repository
    if not ("/" in model_path and not model_path.startswith("/")):
        # It's a local path, check if it exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"BERT model not found at {model_path}")
    
    service = BERTInferenceService(model_path, config)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize BERT service")
    
    return service 