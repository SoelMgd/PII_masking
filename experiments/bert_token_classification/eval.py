#!/usr/bin/env python3
"""
BERT PII Token Classification - Entity-Level Evaluation

This script evaluates the fine-tuned BERT model using proper entity-level metrics
instead of token-level metrics, using the same test split as during training.

Key features:
- Loads the same processed data used during training
- Uses the same train/val/test split as training
- Evaluates on test set using CustomPIIEvaluator for entity-level metrics
- Saves detailed results to results/ directory
"""

import os
import json
import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time
from dataclasses import asdict

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

# Import our evaluation classes
try:
    from pii_masking.text_processing import PIIPrediction, EntitySpan
    from pii_masking.custom_evaluator import CustomPIIEvaluator
    from pii_masking.data_loader import PIIExample
    IMPORTS_AVAILABLE = True
    logger.info("Successfully imported pii_masking modules")
except ImportError as e:
    logger.error(f"Could not import pii_masking modules: {e}")
    IMPORTS_AVAILABLE = False
    sys.exit(1)


def load_processed_data(data_path: str) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    """
    Load processed dataset and label mappings.
    
    Returns:
        Tuple of (examples, label2id, id2label)
    """
    # Load processed examples
    with open(f"{data_path}/processed_examples.json", 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # Load label mappings
    with open(f"{data_path}/label_mappings.json", 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    label2id = mappings['label2id']
    id2label = mappings['id2label']
    # Convert string keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    
    logger.info(f"Loaded {len(examples)} examples")
    logger.info(f"Number of labels: {len(label2id)}")
    
    return examples, label2id, id2label


def split_data(examples: List[Dict], train_split: float, val_split: float, test_split: float):
    """Split data into train/val/test sets using the same logic as training."""
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    n_total = len(examples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    logger.info(f"Data split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    
    return train_examples, val_examples, test_examples


def convert_bert_example_to_pii_example(bert_example: Dict) -> PIIExample:
    """
    Convert BERT processed example to PIIExample format for evaluation.
        
        Args:
        bert_example: Dictionary with tokens, labels, original_text, token_spans
        
    Returns:
        PIIExample object
    """
    # Extract span labels from token-level labels
    span_labels = []
    tokens = bert_example['tokens']
    labels = bert_example['labels']
    token_spans = bert_example['token_spans']
    original_text = bert_example['original_text']
    
    # Convert token-level labels to span labels
    current_entity = None
    current_start = None
    
    for i, (token, label, (start, end)) in enumerate(zip(tokens, labels, token_spans)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if label != 'O':
            if current_entity is None:
                # Start new entity
                current_entity = label
                current_start = start
            elif current_entity == label:
                # Continue current entity
                pass
            else:
                # End previous entity and start new one
                if current_start is not None:
                    span_labels.append([current_start, token_spans[i-1][1], current_entity])
                current_entity = label
                current_start = start
        else:
            # End current entity if exists
            if current_entity is not None:
                span_labels.append([current_start, token_spans[i-1][1], current_entity])
                current_entity = None
                current_start = None
    
    # Handle entity at end of sequence
    if current_entity is not None and current_start is not None:
        # Find the last non-special token
        last_span_idx = len(token_spans) - 1
        while last_span_idx >= 0 and tokens[last_span_idx] in ['[CLS]', '[SEP]', '[PAD]']:
            last_span_idx -= 1
        if last_span_idx >= 0:
            span_labels.append([current_start, token_spans[last_span_idx][1], current_entity])
    
    # Create masked text (simple approach)
    masked_text = original_text
    for start, end, label in reversed(span_labels):  # Reverse to maintain positions
        entity_text = original_text[start:end]
        placeholder = f"[{label}]"
        masked_text = masked_text[:start] + placeholder + masked_text[end:]
    
    return PIIExample(
        unmasked_text=original_text,
        masked_text=masked_text,
        span_labels=span_labels,
        privacy_mask={},
        bio_labels=[],
        tokenised_text=tokens
    )


class BERTTokenClassificationModel:
    """
    BERT Token Classification Model for PII detection evaluation.
    """
    
    def __init__(self, model_path: str):
        """Initialize the BERT model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.device = torch.device("cpu")  # Use CPU for evaluation
        
        logger.info(f"Initialized BERT model with path: {model_path}")
        
    def initialize(self) -> bool:
        """Initialize the model and tokenizer."""
        try:
            # Load model and tokenizer
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Move to CPU and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.is_initialized = True
            logger.info(f"BERT model initialized successfully on {self.device}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            return False
    
    def predict_single(self, text: str) -> PIIPrediction:
        """
        Predict PII entities for a single text.
        
        Args:
            text: Input text
            
        Returns:
            PIIPrediction object with entities, spans, and masked text
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        try:
            # Tokenize input with offset mapping
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
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
            prediction = PIIPrediction(
                entities=entities_dict,
                spans=spans,
                masked_text=masked_text,
                original_text=text
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
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
        """Merge adjacent spans of the same entity type."""
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
        """Determine if two spans should be merged."""
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
    
    def _reconstruct_masked_text(self, text: str, spans: List) -> str:
        """Reconstruct masked text from spans."""
        masked_text = text
        
        # Sort spans by start position in reverse order for replacement
        reverse_sorted_spans = sorted(spans, key=lambda x: x.start, reverse=True)
        
        # Replace each span with placeholder
        entity_counters = {}
        for span in reverse_sorted_spans:
            entity_type = span.entity_type
            if entity_type not in entity_counters:
                entity_counters[entity_type] = 0
            entity_counters[entity_type] += 1
            
            placeholder = f"[{entity_type}_{entity_counters[entity_type]}]"
            masked_text = masked_text[:span.start] + placeholder + masked_text[span.end:]
        
        return masked_text
    

def evaluate_bert_model(model_path: str, data_path: str, results_path: str):
    """
    Evaluate BERT model using entity-level metrics on test set.
    
    Args:
        model_path: Path to the fine-tuned BERT model
        data_path: Path to the processed BERT data
        results_path: Path to save evaluation results
    """
    logger.info("Starting BERT model evaluation with entity-level metrics")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Results path: {results_path}")
    
    # Load processed data
    logger.info("Loading processed data...")
    examples, label2id, id2label = load_processed_data(data_path)
    
    # Split data using same logic as training
    logger.info("Splitting data...")
    train_examples, val_examples, test_examples = split_data(
        examples, train_split=0.8, val_split=0.1, test_split=0.1
    )
    
    logger.info(f"Using {len(test_examples)} test examples for evaluation")
    
    # Initialize BERT model
    logger.info("Initializing BERT model...")
    model = BERTTokenClassificationModel(model_path)
    if not model.initialize():
        logger.error("Failed to initialize BERT model")
        return
    
    # Convert test examples to PIIExample format
    logger.info("Converting test examples to PIIExample format...")
    pii_examples = []
    for i, bert_example in enumerate(test_examples):
        try:
            pii_example = convert_bert_example_to_pii_example(bert_example)
            pii_examples.append(pii_example)
        except Exception as e:
            logger.error(f"Error converting example {i}: {e}")
            continue
    
    logger.info(f"Successfully converted {len(pii_examples)} examples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    failed_predictions = 0
    
    for i, pii_example in enumerate(pii_examples):
        try:
            prediction = model.predict_single(pii_example.unmasked_text)
            predictions.append(prediction)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{len(pii_examples)} predictions")
                    
        except Exception as e:
            logger.error(f"Error predicting example {i}: {e}")
            # Add empty prediction
            predictions.append(PIIPrediction(
                entities={}, 
                spans=[], 
                masked_text=pii_example.unmasked_text,
                original_text=pii_example.unmasked_text
            ))
            failed_predictions += 1
    
    logger.info(f"Generated {len(predictions)} predictions ({failed_predictions} failed)")
    
    # Initialize evaluator and run evaluation
    logger.info("Running entity-level evaluation...")
    evaluator = CustomPIIEvaluator()
    
    evaluation_result = evaluator.evaluate_dataset(
        examples=pii_examples,
        predictions=predictions,
        experiment_name="bert_classic_token_classification_entity_eval",
        model_name="bert-base-uncased-finetuned",
        config={
            "model_path": model_path,
            "data_path": data_path,
            "test_examples_count": len(test_examples),
            "evaluation_type": "entity_level",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1
        }
    )
    
    # Print results
    logger.info("Evaluation completed!")
    print("\n" + "="*80)
    print("BERT CLASSIC TOKEN CLASSIFICATION - ENTITY-LEVEL EVALUATION RESULTS")
    print("="*80)
    print(f"Total examples: {evaluation_result.total_examples}")
    print(f"Successful predictions: {evaluation_result.successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"\nOverall Metrics:")
    print(f"  Precision: {evaluation_result.precision:.4f}")
    print(f"  Recall: {evaluation_result.recall:.4f}")
    print(f"  F1-Score: {evaluation_result.f1_score:.4f}")
    print(f"\nEntity Counts:")
    print(f"  True entities: {evaluation_result.total_true_entities}")
    print(f"  Predicted entities: {evaluation_result.total_pred_entities}")
    print(f"  Correct entities: {evaluation_result.total_correct_entities}")
    
    print(f"\nPer-Class Metrics:")
    for entity_type, metrics in evaluation_result.per_class_metrics.items():
        print(f"  {entity_type}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    # Save results
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "bert_classic_token_classification_entity_eval.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    print(f"\nDetailed results saved to: {results_file}")
    print("="*80)


def main():
    """Main evaluation function."""
    
    print("BERT Classic PII Token Classification - Entity-Level Evaluation")
    print("="*60)
    
    # Configuration
    model_path = "/Users/twin/Documents/pii-masking-200k/models/bert_classic_token_classif"
    data_path = "/Users/twin/Documents/pii-masking-200k/data/processed_data_bert"
    results_path = "/Users/twin/Documents/pii-masking-200k/results"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please ensure you have:")
        print("   1. Completed training with bert_finetuning.py")
        print("   2. Downloaded the model from Kaggle to the models directory")
        return
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"Data not found at {data_path}")
        print("💡 Please ensure the processed BERT data exists")
        return
    
    # Run evaluation
    try:
        evaluate_bert_model(model_path, data_path, results_path)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 