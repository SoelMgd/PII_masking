#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Mistral token classification model.

This script:
1. Loads the fine-tuned token classification model
2. Performs inference on the validation dataset
3. Converts token-level predictions to entity spans
4. Uses the custom evaluator to calculate entity-level metrics
"""

import os
import sys
import json
import pickle
import logging
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pii_masking import PIIExample
from pii_masking.text_processing import EntitySpan, PIIPrediction
from pii_masking.custom_evaluator import evaluate_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenClassificationExample:
    """Example for token classification (matching the training format)."""
    text: str
    tokens: List[str]
    token_ids: List[int]
    labels: List[str]
    token_positions: List[Tuple[int, int]]

class MinistralTokenClassifier(nn.Module):
    """Ministral-8B model with token classification head - Inference version."""
    
    def __init__(self, model_name: str, num_labels: int, device: torch.device):
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        
        print(f"🔄 Loading model: {model_name}")
        
        # Load configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Use float32 for CPU compatibility and remove device_map for CPU
        dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
        model_kwargs = {
            'torch_dtype': dtype,
            'trust_remote_code': True
        }
        
        # Only use device_map if CUDA is available
        if torch.cuda.is_available():
            model_kwargs['device_map'] = "auto"
        
        self.backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        
        # Add classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Keep classifier in same dtype as backbone
        self.classifier = self.classifier.to(dtype=dtype)
        
        self.num_labels = num_labels
        
        print(f"✅ Model initialized with {num_labels} labels")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for inference."""
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Ensure dtype consistency
        hidden_states = hidden_states.to(self.classifier.weight.dtype)
        
        # Classification for each token
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_labels]
        
        return {'logits': logits}

def load_trained_model(model_path: Path, model_name: str, num_labels: int, device: torch.device):
    """Load the trained token classification model."""
    print(f"📂 Loading trained model from {model_path}")
    
    # Initialize model architecture
    model = MinistralTokenClassifier(model_name, num_labels, device)
    model.to(device)
    
    # Load trained weights
    checkpoint_path = model_path / "pytorch_model.bin"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load only the classifier weights (backbone was frozen)
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    print("✅ Model loaded successfully")
    return model, checkpoint

def load_validation_dataset(dataset_path: Path):
    """Load the validation dataset from pickle file."""
    print(f"📂 Loading validation dataset from {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✅ Loaded {len(dataset['texts'])} validation examples")
    return dataset

def convert_to_pii_examples(dataset: Dict[str, Any]) -> List[PIIExample]:
    """
    Convert the token classification dataset back to PIIExample format.
    
    This is a reverse conversion from the processing done in dataset_processing_token.py
    """
    pii_examples = []
    
    texts = dataset['texts']
    tokens_list = dataset['tokens'] 
    labels_list = dataset['labels']
    id_to_label = dataset['id_to_label']
    
    print(f"🔄 Converting {len(texts)} examples to PIIExample format...")
    
    for i, (text, tokens, labels) in enumerate(zip(texts, tokens_list, labels_list)):
        try:
            # Reconstruct span_labels from token labels
            span_labels = []
            current_entity = None
            start_pos = 0
            
            # Calculate token positions by reconstructing from tokens
            token_positions = []
            current_pos = 0
            for token in tokens:
                # Find token in text starting from current position
                token_start = text.find(token, current_pos)
                if token_start == -1:
                    token_start = current_pos
                token_end = token_start + len(token)
                token_positions.append((token_start, token_end))
                current_pos = token_end
            
            # Convert token labels to span labels
            for j, (label, (token_start, token_end)) in enumerate(zip(labels, token_positions)):
                if label != 'O':  # Non-O label
                    if current_entity is None:
                        # Start of new entity
                        current_entity = {
                            'start': token_start,
                            'end': token_end,
                            'label': label
                        }
                    elif current_entity['label'] == label:
                        # Continue current entity
                        current_entity['end'] = token_end
                    else:
                        # End current entity and start new one
                        span_labels.append([current_entity['start'], current_entity['end'], current_entity['label']])
                        current_entity = {
                            'start': token_start,
                            'end': token_end,
                            'label': label
                        }
                else:  # O label
                    if current_entity is not None:
                        # End current entity
                        span_labels.append([current_entity['start'], current_entity['end'], current_entity['label']])
                        current_entity = None
            
            # Don't forget the last entity
            if current_entity is not None:
                span_labels.append([current_entity['start'], current_entity['end'], current_entity['label']])
            
            # Create PIIExample
            pii_example = PIIExample(
                masked_text=text,  # We don't have the original masked text, so use unmasked
                unmasked_text=text,
                privacy_mask={},  # We don't have this from token classification data
                span_labels=span_labels,
                bio_labels=labels,
                tokenised_text=tokens
            )
            
            pii_examples.append(pii_example)
            
        except Exception as e:
            logger.warning(f"Error converting example {i}: {e}")
            continue
    
    print(f"✅ Converted {len(pii_examples)} examples successfully")
    return pii_examples

def predict_batch(model: MinistralTokenClassifier, batch_data: Dict[str, torch.Tensor], 
                 id_to_label: Dict[int, str], device: torch.device) -> List[List[str]]:
    """Predict labels for a batch of examples."""
    model.eval()
    
    with torch.no_grad():
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch_data.items()}
        
        # Forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs['logits']  # [batch_size, seq_len, num_labels]
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        attention_mask = batch['attention_mask']
        
        # Convert to labels
        batch_predictions = []
        for i in range(predictions.shape[0]):
            seq_predictions = []
            seq_length = attention_mask[i].sum().item()  # Get actual sequence length
            
            for j in range(seq_length):
                label_id = predictions[i, j].item()
                label = id_to_label.get(label_id, 'O')
                seq_predictions.append(label)
            
            batch_predictions.append(seq_predictions)
    
    return batch_predictions

def run_inference(model: MinistralTokenClassifier, dataset: Dict[str, Any], 
                 device: torch.device, batch_size: int = 16, max_length: int = 96) -> List[List[str]]:
    """Run inference on the entire dataset."""
    print(f"🚀 Running inference on {len(dataset['texts'])} examples...")
    
    all_predictions = []
    id_to_label = dataset['id_to_label']
    
    # Prepare data for batched inference
    num_examples = len(dataset['texts'])
    
    for start_idx in tqdm(range(0, num_examples, batch_size), desc="Inference"):
        end_idx = min(start_idx + batch_size, num_examples)
        
        # Prepare batch
        batch_token_ids = []
        batch_attention_masks = []
        
        for i in range(start_idx, end_idx):
            token_ids = dataset['token_ids'][i]
            
            # Pad/truncate to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids + [0] * (max_length - len(token_ids))
            
            attention_mask = [1 if tid != 0 else 0 for tid in token_ids]
            
            batch_token_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)
        
        # Convert to tensors
        batch_data = {
            'input_ids': torch.tensor(batch_token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_masks, dtype=torch.long)
        }
        
        # Predict
        batch_predictions = predict_batch(model, batch_data, id_to_label, device)
        all_predictions.extend(batch_predictions)
    
    print(f"✅ Inference completed on {len(all_predictions)} examples")
    return all_predictions

def convert_predictions_to_spans(texts: List[str], tokens_list: List[List[str]], 
                               predictions: List[List[str]]) -> List[PIIPrediction]:
    """
    Convert token-level predictions to entity spans for the evaluator.
    """
    print(f"🔄 Converting {len(predictions)} predictions to entity spans...")
    
    pii_predictions = []
    
    for text, tokens, pred_labels in zip(texts, tokens_list, predictions):
        try:
            # Calculate token positions
            token_positions = []
            current_pos = 0
            
            for token in tokens:
                # Find token in text starting from current position
                token_start = text.find(token, current_pos)
                if token_start == -1:
                    token_start = current_pos
                token_end = token_start + len(token)
                token_positions.append((token_start, token_end))
                current_pos = token_end
            
            # Convert token predictions to entity spans
            spans = []
            entities = defaultdict(list)
            current_entity = None
            
            for i, (label, (token_start, token_end)) in enumerate(zip(pred_labels, token_positions)):
                if label != 'O':  # Non-O label
                    if current_entity is None or current_entity['label'] != label:
                        # End previous entity if exists
                        if current_entity is not None:
                            entity_text = text[current_entity['start']:current_entity['end']]
                            span = EntitySpan(
                                entity_type=current_entity['label'],
                                start=current_entity['start'],
                                end=current_entity['end'],
                                text=entity_text
                            )
                            spans.append(span)
                            entities[current_entity['label']].append(entity_text)
                        
                        # Start new entity
                        current_entity = {
                            'start': token_start,
                            'end': token_end,
                            'label': label
                        }
                    else:
                        # Continue current entity
                        current_entity['end'] = token_end
                else:  # O label
                    if current_entity is not None:
                        # End current entity
                        entity_text = text[current_entity['start']:current_entity['end']]
                        span = EntitySpan(
                            entity_type=current_entity['label'],
                            start=current_entity['start'],
                            end=current_entity['end'],
                            text=entity_text
                        )
                        spans.append(span)
                        entities[current_entity['label']].append(entity_text)
                        current_entity = None
            
            # Don't forget the last entity
            if current_entity is not None:
                entity_text = text[current_entity['start']:current_entity['end']]
                span = EntitySpan(
                    entity_type=current_entity['label'],
                    start=current_entity['start'],
                    end=current_entity['end'],
                    text=entity_text
                )
                spans.append(span)
                entities[current_entity['label']].append(entity_text)
            
            # Create PIIPrediction
            prediction = PIIPrediction(
                entities=dict(entities),
                spans=spans,
                masked_text="",  # Not needed for evaluation
                original_text=text
            )
            
            pii_predictions.append(prediction)
            
        except Exception as e:
            logger.warning(f"Error converting prediction: {e}")
            # Create empty prediction as fallback
            prediction = PIIPrediction(
                entities={},
                spans=[],
                masked_text="",
                original_text=text
            )
            pii_predictions.append(prediction)
    
    print(f"✅ Converted {len(pii_predictions)} predictions to spans")
    return pii_predictions

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral token classification model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="data/llm_token_classif/val_dataset.pkl",
        help="Path to validation dataset pickle file"
    )
    parser.add_argument(
        "--model-name", 
        type=str,
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Base model name used for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save evaluation results"
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed per-example results"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load validation dataset
    val_dataset_path = Path(args.val_dataset)
    if not val_dataset_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_dataset_path}")
    
    dataset = load_validation_dataset(val_dataset_path)
    
    # Load trained model
    model_path = Path(args.model_path)
    num_labels = dataset['num_labels']
    
    model, checkpoint = load_trained_model(model_path, args.model_name, num_labels, device)
    
    # Convert dataset to PIIExample format (ground truth)
    pii_examples = convert_to_pii_examples(dataset)
    
    # Run inference
    predictions = run_inference(model, dataset, device, args.batch_size, args.max_length)
    
    # Convert predictions to spans
    pii_predictions = convert_predictions_to_spans(
        dataset['texts'], 
        dataset['tokens'], 
        predictions
    )
    
    # Evaluate using custom evaluator
    print(f"\n🔍 Running entity-level evaluation...")
    
    config = {
        'model_name': args.model_name,
        'model_path': str(model_path),
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'num_labels': num_labels
    }
    
    # Add training config if available
    if 'config' in checkpoint:
        config.update(checkpoint['config'])
    
    result = evaluate_predictions(
        examples=pii_examples,
        predictions=pii_predictions,
        experiment_name="mistral_token_classification_eval",
        model_name="Ministral-8B-Token-Classifier",
        config=config,
        output_file=args.output_file,
        save_details=args.save_details,
        print_results=True
    )
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📊 Overall F1-Score: {result.f1_score:.4f}")
    print(f"📊 Precision: {result.precision:.4f}")
    print(f"📊 Recall: {result.recall:.4f}")
    print(f"📊 Total entities: {result.total_true_entities}")
    print(f"📊 Correct entities: {result.total_correct_entities}")
    
    if args.output_file:
        print(f"💾 Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 