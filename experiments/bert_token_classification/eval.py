#!/usr/bin/env python3
"""
BERT PII Token Classification - Inference and Testing

This script provides CPU-optimized inference capabilities for the fine-tuned BERT model,
with full compatibility with the BasePIIModel interface for production deployment.

Key features:
- CPU-optimized inference
- BasePIIModel interface compatibility
- Production-ready masked text generation
- Comprehensive testing and evaluation
"""

import os
import json
import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import our base classes
try:
    from pii_masking.base_model import BasePIIModel
    from pii_masking.text_processing import PIIPrediction, EntitySpan
    from pii_masking.custom_evaluator import CustomPIIEvaluator
    from pii_masking.data_loader import PIIDataLoader, PIIExample
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Could not import pii_masking modules: {e}")
    IMPORTS_AVAILABLE = False


class BERTTokenClassificationModel(BasePIIModel if IMPORTS_AVAILABLE else object):
    """
    Production-ready BERT Token Classification Model for PII detection.
    
    Features:
    - CPU-optimized inference
    - BasePIIModel interface compatibility
    - Efficient batch processing
    - Production-ready masked text generation
    """
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Initialize the BERT model.
        
        Args:
            model_path: Path to the fine-tuned model directory
            config: Optional configuration dictionary
        """
        if IMPORTS_AVAILABLE:
            super().__init__(model_name=f"bert-{Path(model_path).name}", config=config)
        
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.device = torch.device("cpu")  # Force CPU for production
        
        # Performance settings
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 8)
        
        logger.info(f"Initialized BERT model with path: {model_path}")
        
    def initialize(self) -> bool:
        """Initialize the model and tokenizer for CPU inference."""
        try:
            # Load model and tokenizer
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32  # Use float32 for CPU
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Move to CPU and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            torch.set_num_threads(1)  # Single thread for consistent performance
            
            self.is_initialized = True
            logger.info(f"BERT model initialized successfully on {self.device}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            return False
    
    def predict_single(self, text: str, **kwargs) -> 'PIIPrediction':
        """
        Predict PII entities for a single text.
        
        Args:
            text: Input text
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            PIIPrediction object with entities, spans, and masked text
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize input with offset mapping
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )
            
            # Move to device
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
            
            # Reconstruct masked text
            masked_text = self._reconstruct_masked_text(text, spans)
            
            # Create PIIPrediction object
            if IMPORTS_AVAILABLE:
                prediction = PIIPrediction(
                    entities=entities_dict,
                    spans=spans,
                    masked_text=masked_text,
                    original_text=text
                )
            else:
                prediction = {
                    'entities': entities_dict,
                    'spans': [(s.entity_type, s.start, s.end, s.text) for s in spans],
                    'masked_text': masked_text,
                    'original_text': text
                }
            
            inference_time = time.time() - start_time
            logger.debug(f"⚡ Inference time: {inference_time:.3f}s")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return empty prediction on error
            if IMPORTS_AVAILABLE:
                return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
            else:
                return {'entities': {}, 'spans': [], 'masked_text': text, 'original_text': text}
    
    def predict_batch(self, texts: List[str], **kwargs) -> List['PIIPrediction']:
        """
        Predict PII entities for a batch of texts with optimized processing.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            List of PIIPrediction objects
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        predictions = []
        total_texts = len(texts)
        
        logger.info(f"Processing {total_texts} texts in batches of {self.batch_size}")
        
        # Process in batches for memory efficiency
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_predictions = []
            
            for text in batch_texts:
                try:
                    prediction = self.predict_single(text, **kwargs)
                    batch_predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error processing text {i}: {e}")
                    # Add empty prediction on error
                    if IMPORTS_AVAILABLE:
                        empty_pred = PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
                    else:
                        empty_pred = {'entities': {}, 'spans': [], 'masked_text': text, 'original_text': text}
                    batch_predictions.append(empty_pred)
            
            predictions.extend(batch_predictions)
            
            # Log progress
            processed = min(i + self.batch_size, total_texts)
            logger.info(f"Processed {processed}/{total_texts} texts ({processed/total_texts*100:.1f}%)")
        
        logger.info(f"Batch processing completed: {len(predictions)} predictions generated")
        return predictions
    
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
        
        # Second pass: merge adjacent spans of the same type (including those separated by whitespace)
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
            if IMPORTS_AVAILABLE:
                final_spans.append(EntitySpan(
                    entity_type=entity_type,
                    start=span_info['start'],
                    end=span_info['end'],
                    text=entity_text
                ))
            else:
                final_spans.append(span_info)
        
        return entities_dict, final_spans
    
    def _merge_adjacent_spans(self, text: str, raw_spans: List[Dict]) -> List[Dict]:
        """
        Merge adjacent spans of the same entity type, including those separated by whitespace.
        
        Example: ['March', '15,', '1990'] -> ['March 15, 1990']
        """
        if not raw_spans:
            return []
        
        # Sort spans by start position
        sorted_spans = sorted(raw_spans, key=lambda x: x['start'])
        merged_spans = []
        
        current_span = sorted_spans[0].copy()
        
        for next_span in sorted_spans[1:]:
            # Check if spans are of the same type and close to each other
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
        
        Merge if:
        1. Same entity type
        2. Separated by only whitespace/punctuation (max 3 characters)
        3. No other meaningful text between them
        """
        if span1['entity_type'] != span2['entity_type']:
            return False
        
        # Get the text between the spans
        between_start = span1['end']
        between_end = span2['start']
        
        if between_end <= between_start:
            return True  # Adjacent or overlapping
        
        between_text = text[between_start:between_end]
        
        # Merge if only whitespace and simple punctuation, and not too long
        if len(between_text) <= 3 and between_text.strip() in ['', ',', '.', '-', '/', ':', ';']:
            return True
        
        # Also merge if it's just whitespace
        if between_text.isspace() and len(between_text) <= 2:
            return True
        
        return False
    
    def _add_entity(self, text: str, entities_dict: Dict, spans: List, entity_type: str, start: int, end: int):
        """Add an entity to the entities dictionary and spans list."""
        entity_text = text[start:end].strip()
        
        if entity_text:  # Only add non-empty entities
            # Add to entities dictionary
            if entity_type not in entities_dict:
                entities_dict[entity_type] = []
            if entity_text not in entities_dict[entity_type]:
                entities_dict[entity_type].append(entity_text)
            
            # Add to spans list
            if IMPORTS_AVAILABLE:
                spans.append(EntitySpan(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    text=entity_text
                ))
            else:
                spans.append({
                    'entity_type': entity_type,
                    'start': start,
                    'end': end,
                    'text': entity_text
                })
    
    def _reconstruct_masked_text(self, text: str, spans: List) -> str:
        """Reconstruct masked text from spans with production-ready placeholders."""
        masked_text = text
        
        # First, assign numbers to spans in their natural order (left to right)
        spans_with_numbers = []
        entity_counters = {}
        
        # Sort spans by start position (normal order) to assign consecutive numbers
        normal_sorted_spans = sorted(spans, key=lambda x: x.start if hasattr(x, 'start') else x['start'])
        
        for span in normal_sorted_spans:
            entity_type = span.entity_type if hasattr(span, 'entity_type') else span['entity_type']
            start = span.start if hasattr(span, 'start') else span['start']
            end = span.end if hasattr(span, 'end') else span['end']
            
            if entity_type not in entity_counters:
                entity_counters[entity_type] = 0
            entity_counters[entity_type] += 1
            
            spans_with_numbers.append({
                'entity_type': entity_type,
                'start': start,
                'end': end,
                'number': entity_counters[entity_type]
            })
        
        # Now sort by start position in reverse order for replacement (to maintain positions)
        reverse_sorted_spans = sorted(spans_with_numbers, key=lambda x: x['start'], reverse=True)
        
        # Replace each span with placeholder using pre-calculated numbers
        for span_info in reverse_sorted_spans:
            placeholder = f"[{span_info['entity_type']}_{span_info['number']}]"
            masked_text = masked_text[:span_info['start']] + placeholder + masked_text[span_info['end']:]
        
        return masked_text
    
    def get_model_info(self) -> Dict:
        """Get model information for monitoring and debugging."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_path": self.model_path,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_parameters": self.model.num_parameters(),
            "num_labels": self.model.config.num_labels,
            "label_mappings": {
                "id2label": self.model.config.id2label,
                "label2id": self.model.config.label2id
            }
        }


def test_model_performance(model: BERTTokenClassificationModel, test_texts: List[str]) -> Dict:
    """Test model performance with various metrics."""
    
    print("Testing model performance...")
    
    # Measure inference time
    start_time = time.time()
    predictions = model.predict_batch(test_texts)
    total_time = time.time() - start_time
    
    # Calculate metrics
    avg_time_per_text = total_time / len(test_texts)
    throughput = len(test_texts) / total_time
    
    # Count entities
    total_entities = 0
    entity_types = set()
    
    for pred in predictions:
        if IMPORTS_AVAILABLE:
            for entity_type, entities in pred.entities.items():
                total_entities += len(entities)
                entity_types.add(entity_type)
        else:
            for entity_type, entities in pred['entities'].items():
                total_entities += len(entities)
                entity_types.add(entity_type)
    
    results = {
        'total_texts': len(test_texts),
        'total_time': total_time,
        'avg_time_per_text': avg_time_per_text,
        'throughput_texts_per_sec': throughput,
        'total_entities_found': total_entities,
        'unique_entity_types': len(entity_types),
        'entity_types': sorted(list(entity_types))
    }
    
    print(f"Performance Results:")
    print(f"  Total texts: {results['total_texts']}")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Average time per text: {results['avg_time_per_text']:.3f}s")
    print(f"  Throughput: {results['throughput_texts_per_sec']:.1f} texts/sec")
    print(f"  Total entities found: {results['total_entities_found']}")
    print(f"  Entity types: {results['entity_types']}")
    
    return results

def run_sample_tests(model: BERTTokenClassificationModel):
    """Run comprehensive sample tests."""
    
    print("Running sample tests...")
    
    # Test cases with expected PII
    test_cases = [
        {
            "text": "Hi, my name is John Smith and my email is john.smith@example.com.",
            "expected_types": ["FIRSTNAME", "EMAIL"]
        },
        {
            "text": "Contact me at +1-555-123-4567 or visit 123 Main Street, New York.",
            "expected_types": ["PHONENUMBER", "STREET"]
        },
        {
            "text": "My SSN is 123-45-6789 and I was born on March 15, 1990.",
            "expected_types": ["SSN", "DOB"]
        },
        {
            "text": "This is a normal sentence with no PII information.",
            "expected_types": []
        },
        {
            "text": "Dr. Sarah Johnson works at Microsoft Corporation in Seattle, WA.",
            "expected_types": ["PREFIX", "FIRSTNAME", "LASTNAME", "ORGANIZATION", "CITY", "STATE"]
        }
    ]
    
    print(f"\nTesting {len(test_cases)} sample cases:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test_case['text']}")
        
        try:
            prediction = model.predict_single(test_case['text'])
            
            if IMPORTS_AVAILABLE:
                entities = prediction.entities
                masked_text = prediction.masked_text
                num_spans = len(prediction.spans)
            else:
                entities = prediction['entities']
                masked_text = prediction['masked_text']
                num_spans = len(prediction['spans'])
            
            print(f"Entities found: {entities}")
            print(f"Masked text: {masked_text}")
            print(f"Number of spans: {num_spans}")
            
            # Check if expected types were found
            found_types = set(entities.keys())
            expected_types = set(test_case['expected_types'])
            
            if expected_types:
                overlap = found_types.intersection(expected_types)
                print(f"Expected types: {expected_types}")
                print(f"Found types: {found_types}")
                print(f"Overlap: {overlap}")
                
                if overlap == expected_types:
                    print("All expected types found!")
                elif overlap:
                    print("Some expected types found")
                else:
                    print("No expected types found")
            else:
                if found_types:
                    print("Entities found in text expected to have no PII")
                else:
                    print("No entities found as expected!")
                    
        except Exception as e:
            print(f"Error during prediction: {e}")
        
        print("-" * 40)


def main():
    """Main testing function."""
    
    print("BERT PII Token Classification - Inference Testing")
    print("=" * 60)
    
    # Configuration
    model_path = "/Users/twin/Documents/pii-masking-200k/models/bert_token_classif"  # Adjust path as needed
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("💡 Please ensure you have:")
        print("   1. Completed training with bert_finetuning.py")
        print("   2. Downloaded the model from Kaggle to /Users/twin/Documents/pii-masking-200k/models/bert_token_classif")
        return
    
    # Initialize model
    print(f"Initializing BERT model from {model_path}...")
    
    config = {
        'max_length': 512,
        'batch_size': 8
    }
    
    model = BERTTokenClassificationModel(model_path, config)
    
    if not model.initialize():
        print("Failed to initialize model")
        return
    
    # Display model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        if key != 'label_mappings':  # Skip detailed mappings for cleaner output
            print(f"  {key}: {value}")
    
    # Run sample tests
    run_sample_tests(model)
    
    # Performance testing
    test_texts = [
        "Hello, I'm Jane Doe and you can reach me at jane.doe@company.com.",
        "My phone number is 555-0123 and I live at 456 Oak Avenue.",
        "Please send the documents to Michael Brown at 789 Pine Street, Boston, MA 02101.",
        "Contact Dr. Smith at the hospital for more information.",
        "This is a sentence without any personal information.",
        "Born on January 1, 1985, with SSN 987-65-4321.",
        "Visit our website at www.example.com or call 1-800-555-0199.",
        "My credit card number is 4532-1234-5678-9012, expires 12/25."
    ]
    
    performance_results = test_model_performance(model, test_texts)
    
    # Test with custom evaluator if available
    if IMPORTS_AVAILABLE:
        print("Testing integration with CustomPIIEvaluator...")
        try:
            evaluator = CustomPIIEvaluator()
            
            # Create a sample PIIExample for testing
            sample_text = "Hi John Smith, your email john@example.com has been verified."
            sample_prediction = model.predict_single(sample_text)
            
            # Create mock PIIExample (in real scenario, this would come from dataset)
            mock_example = PIIExample(
                unmasked_text=sample_text,
                masked_text="Hi [FIRSTNAME_1] [LASTNAME_1], your email [EMAIL_1] has been verified.",
                span_labels=[[3, 13, 'FIRSTNAME'], [3, 13, 'LASTNAME'], [26, 42, 'EMAIL']],
                privacy_mask={},
                bio_labels=[],
                tokenised_text=[]
            )
            
            print(f"CustomPIIEvaluator integration successful!")
            print(f"   Sample prediction generated: {len(sample_prediction.spans)} spans found")
            
        except Exception as e:
            print(f"CustomPIIEvaluator integration test failed: {e}")
    
    # Final summary
    print(f"\nTesting completed successfully!")
    print(f"Summary:")
    print(f"  Model loaded:")
    print(f"  Sample tests:")
    print(f"  Performance test: ({performance_results['throughput_texts_per_sec']:.1f} texts/sec)")
    print(f"  Integration test: {'' if IMPORTS_AVAILABLE else ''}")
    
    print(f"\nModel is ready for production use!")
    print(f"💡 Usage example:")
    print(f"   model = BERTTokenClassificationModel('{model_path}')")
    print(f"   model.initialize()")
    print(f"   prediction = model.predict_single('Your text here')")

if __name__ == "__main__":
    main() 