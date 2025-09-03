#!/usr/bin/env python3
"""
Kaggle fine-tuning script for Ministral-8B token classification.

This script:
1. Loads pre-trained Ministral-8B model and freezes weights
2. Adds a token classification head
3. Trains only the classification head on PII detection
4. Saves the trained model for inference

Designed to run on Kaggle with GPU acceleration.
Structure: Each section marked with ### can be copied to a separate notebook cell.
"""

# =============================================================================
# ### CELL 1: IMPORTS AND SETUP
# =============================================================================

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# ### CELL 2: CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration optimized for Kaggle."""
    model_name: str = "mistralai/Ministral-8B-Instruct-2410"  # Updated model
    max_length: int = 96   # Optimized length
    batch_size: int = 12   # Conservative batch size
    learning_rate: float = 1e-5  # Adjusted for batch_size=12
    num_epochs: int = 1    # Start with 1 epoch due to time constraints
    warmup_steps: int = 150  # Warmup for stability
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 3  # Effective batch = 12*3 = 36
    max_grad_norm: float = 0.5  # Conservative clipping
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    output_dir: str = "./ministral_token_classifier"
    seed: int = 42

# =============================================================================
# ### ALTERNATIVE: FAST TRAINING CONFIG (uncomment to use)
# =============================================================================

# @dataclass
# class TrainingConfig:
#     """Fast training configuration for quick experiments."""
#     model_name: str = "mistralai/Ministral-8B-Instruct-2410"
#     max_length: int = 64   # Even shorter sequences
#     batch_size: int = 24   # Larger batch for speed
#     learning_rate: float = 3e-5  # Higher LR for faster convergence
#     num_epochs: int = 1
#     warmup_steps: int = 100
#     weight_decay: float = 0.01
#     gradient_accumulation_steps: int = 2  # Effective batch = 48
#     max_grad_norm: float = 1.0
#     save_steps: int = 200  # Save more frequently
#     eval_steps: int = 100  # Evaluate more frequently
#     logging_steps: int = 25
#     output_dir: str = "./ministral_token_classifier"
#     seed: int = 42
#     # Use with 15k-20k examples for ~2h training

# Initialize config
config = TrainingConfig()
print(f"📋 Configuration:")
print(f"   Model: {config.model_name}")
print(f"   Batch size: {config.batch_size}")
print(f"   Learning rate: {config.learning_rate}")
print(f"   Max length: {config.max_length}")

# =============================================================================
# ### CELL 3: DATASET CLASS (SIMPLIFIED)
# =============================================================================

class PIITokenDataset(Dataset):
    """PyTorch Dataset for PII token classification - Simplified version."""
    
    def __init__(self, dataset: Dict[str, Any], max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            dataset: Processed dataset dictionary (already tokenized with Tekken v3)
            max_length: Maximum sequence length
        """
        self.texts = dataset['texts']
        self.tokens = dataset['tokens']  # Already tokenized with Tekken v3
        self.labels = dataset['labels']
        self.label_ids = dataset['label_ids']
        self.label_to_id = dataset['label_to_id']
        self.id_to_label = dataset['id_to_label']
        self.max_length = max_length
        
        # Initialize Mistral Tekken v3 tokenizer for consistency
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            self.tokenizer = MistralTokenizer.v3(is_tekken=True)
            print("✅ Mistral Tekken v3 tokenizer initialized")
        except ImportError:
            raise ImportError("Please install mistral-common: pip install mistral-common")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single example - simplified since alignment is already done."""
        text = self.texts[idx]
        original_tokens = self.tokens[idx]  # Already Tekken v3 tokens
        original_labels = self.label_ids[idx]  # Already aligned
        
        # Re-tokenize with Tekken v3 to get token IDs
        token_ids = self._tokenize_to_ids(text)
        
        # Align our pre-computed labels with the token IDs
        aligned_labels = self._align_precomputed_labels(
            original_tokens, original_labels, token_ids, text
        )
        
        # Pad/truncate to max_length
        input_ids = self._pad_or_truncate(token_ids, self.max_length, pad_value=0)
        labels = self._pad_or_truncate(aligned_labels, self.max_length, pad_value=-100)
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _tokenize_to_ids(self, text: str) -> List[int]:
        """Tokenize text to token IDs using Tekken v3."""
        try:
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            
            request = ChatCompletionRequest(
                messages=[UserMessage(content=text)],
                model="ministral"
            )
            
            encoded_result = self.tokenizer.encode_chat_completion(request)
            token_ids = encoded_result.tokens
            
            # Remove the first 2 empty tokens that Tekken always adds
            if len(token_ids) >= 2:
                return token_ids[2:]
            return token_ids
            
        except Exception as e:
            logger.warning(f"Error tokenizing with Tekken: {e}")
            # Fallback: create dummy token IDs
            return [1] * len(text.split())
    
    def _align_precomputed_labels(self, original_tokens: List[str], 
                                 original_labels: List[int], 
                                 token_ids: List[int], 
                                 text: str) -> List[int]:
        """
        Align pre-computed labels with token IDs.
        Since both use Tekken v3, this should be straightforward.
        """
        # Decode token IDs back to tokens for alignment
        try:
            decoded_tokens = []
            for token_id in token_ids:
                token_text = self.tokenizer.decode([token_id])
                decoded_tokens.append(token_text)
            
            # If lengths match, use labels directly
            if len(decoded_tokens) == len(original_tokens):
                return original_labels[:len(token_ids)]
            
            # Otherwise, align based on text reconstruction
            return self._align_by_text_position(
                original_tokens, original_labels, decoded_tokens, text
            )
            
        except Exception as e:
            logger.warning(f"Error in label alignment: {e}")
            # Fallback: pad with "O" labels
            return [0] * len(token_ids)
    
    def _align_by_text_position(self, original_tokens: List[str], 
                               original_labels: List[int],
                               new_tokens: List[str], 
                               text: str) -> List[int]:
        """Align labels based on character positions in text."""
        # Create character-level label mapping
        char_labels = [0] * len(text)
        current_pos = 0
        
        for token, label_id in zip(original_tokens, original_labels):
            token_start = text.find(token, current_pos)
            if token_start != -1:
                token_end = token_start + len(token)
                for i in range(token_start, min(token_end, len(text))):
                    char_labels[i] = label_id
                current_pos = token_end
        
        # Map to new tokens
        aligned_labels = []
        current_pos = 0
        
        for token in new_tokens:
            token_start = text.find(token, current_pos)
            if token_start != -1:
                token_end = token_start + len(token)
                # Use the most common label in this span
                span_labels = char_labels[token_start:token_end]
                if span_labels:
                    label = max(set(span_labels), key=span_labels.count)
                    aligned_labels.append(label)
                else:
                    aligned_labels.append(0)
                current_pos = token_end
            else:
                aligned_labels.append(0)
        
        return aligned_labels
    
    def _pad_or_truncate(self, sequence: List[int], max_length: int, pad_value: int) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_value] * (max_length - len(sequence))

print("✅ Dataset class defined")

# =============================================================================
# ### CELL 4: MODEL DEFINITION
# =============================================================================

class MinistralTokenClassifier(nn.Module):
    """Ministral-8B model with token classification head."""
    
    def __init__(self, model_name: str, num_labels: int, freeze_backbone: bool = True):
        """
        Initialize the model.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of classification labels
            freeze_backbone: Whether to freeze the backbone model
        """
        super().__init__()
        
        print(f"🔄 Loading model: {model_name}")
        
        # Load configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto",
            trust_remote_code=True  # Required for Ministral
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Backbone model frozen")
        
        # Add classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Keep classifier in float32 for numerical stability
        # self.classifier = self.classifier.half()  # REMOVED - cause of NaN!
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        self.num_labels = num_labels
        
        print(f"✅ Model initialized with {num_labels} labels")
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout and classification
        hidden_states = self.dropout(hidden_states)
        # Convert to float32 before classifier to avoid NaN
        hidden_states_f32 = hidden_states.float()
        logits = self.classifier(hidden_states_f32)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Convert logits to float32 for loss calculation (CrossEntropyLoss expects float32)
            logits_for_loss = logits.float()
            
            # Clamp logits to prevent extreme values
            logits_for_loss = torch.clamp(logits_for_loss, min=-10.0, max=10.0)
            
            # Filter out invalid labels (>= num_labels)
            labels_flat = labels.view(-1)
            valid_mask = (labels_flat >= 0) & (labels_flat < self.num_labels)
            
            if valid_mask.sum() > 0:  # Only calculate loss if we have valid labels
                valid_logits = logits_for_loss.view(-1, self.num_labels)[valid_mask]
                valid_labels = labels_flat[valid_mask]
                
                # Calculate loss only on valid labels
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                loss = loss_fct(valid_logits, valid_labels)
            else:
                # No valid labels, return zero loss
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Check for NaN loss and replace with a large but finite value
            if torch.isnan(loss):
                print("⚠️ NaN loss detected, replacing with 10.0")
                loss = torch.tensor(10.0, device=loss.device, requires_grad=True)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }

print("✅ Model class defined")

# =============================================================================
# ### CELL 5: LOAD DATASET
# =============================================================================

# Load training dataset
train_dataset_path = "/kaggle/input/your-dataset/train_dataset.pkl"  # Update path
val_dataset_path = "/kaggle/input/your-dataset/val_dataset.pkl"      # Update path

print("📂 Loading datasets...")

# Load training data
with open(train_dataset_path, 'rb') as f:
    train_data = pickle.load(f)

train_dataset = PIITokenDataset(train_data, config.max_length)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True,
    num_workers=2
)

# Load validation data if available
val_dataloader = None
if os.path.exists(val_dataset_path):
    with open(val_dataset_path, 'rb') as f:
        val_data = pickle.load(f)
    
    val_dataset = PIITokenDataset(val_data, config.max_length)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2
    )
    print(f"✅ Loaded {len(val_dataset)} validation examples")

# Store label mappings
label_to_id = train_data['label_to_id']
id_to_label = train_data['id_to_label']
num_labels = train_data['num_labels']

print(f"✅ Loaded {len(train_dataset)} training examples")
print(f"🏷️  Number of labels: {num_labels}")
print(f"📊 Labels: {list(label_to_id.keys())}")

# =============================================================================
# ### CELL 6: INITIALIZE MODEL AND OPTIMIZER
# =============================================================================

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Initialize model
model = MinistralTokenClassifier(
    model_name=config.model_name,
    num_labels=num_labels,
    freeze_backbone=True
)

model.to(device)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"📊 Trainable parameters: {trainable_params:,}")
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable ratio: {trainable_params/total_params:.2%}")

# Setup optimizer (only for trainable parameters)
optimizer_params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(
    optimizer_params,
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# Calculate total training steps
total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=total_steps
)

print(f"🚀 Total training steps: {total_steps}")
print(f"🔥 Ready to train!")

# =============================================================================
# ### CELL 7: TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, config, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs['loss']
        
        # Backward pass
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update weights
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Logging
        if step % config.logging_steps == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches

def evaluate(model, val_dataloader, id_to_label, device):
    """Evaluate on validation set."""
    if not val_dataloader:
        print("⚠️  No validation dataset available")
        return {}
    
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Flatten and filter out ignored labels
            batch_labels = batch['labels'].cpu().numpy().flatten()
            batch_predictions = predictions.cpu().numpy().flatten()
            
            # Only keep non-ignored labels
            mask = batch_labels != -100
            all_labels.extend(batch_labels[mask])
            all_predictions.extend(batch_predictions[mask])
    
    # Calculate metrics
    avg_loss = total_loss / len(val_dataloader)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Classification report
    target_names = [id_to_label[i] for i in range(len(id_to_label))]
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    print(f"📊 Validation Loss: {avg_loss:.4f}")
    print(f"📊 Validation F1-Score: {f1:.4f}")
    
    return {
        'loss': avg_loss,
        'f1_score': f1,
        'classification_report': report
    }

def save_model(model, config, label_to_id, id_to_label, output_dir):
    """Save the trained model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict (only classifier since backbone is frozen)
    model_state = {
        'classifier_state_dict': model.classifier.state_dict(),
        'config': config.__dict__,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_labels': len(label_to_id),
        'model_name': config.model_name
    }
    
    torch.save(model_state, output_path / "pytorch_model.bin")
    
    # Save config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"💾 Model saved to {output_path}")

print("✅ Training functions defined")

# =============================================================================
# ### CELL 8: TRAINING LOOP
# =============================================================================

print("🚀 Starting training...")

best_f1 = 0
output_dir = config.output_dir

for epoch in range(config.num_epochs):
    print(f"\n🔄 Starting epoch {epoch + 1}/{config.num_epochs}")
    
    # Train
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch, config, device)
    print(f"✅ Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
    
    # Evaluate
    eval_results = evaluate(model, val_dataloader, id_to_label, device)
    
    if eval_results and eval_results['f1_score'] > best_f1:
        best_f1 = eval_results['f1_score']
        save_model(model, config, label_to_id, id_to_label, output_dir)
        print(f"🎉 New best F1-Score: {best_f1:.4f} - Model saved")

print(f"\n🎉 Training completed!")
print(f"🏆 Best F1-Score: {best_f1:.4f}")
print(f"💾 Model saved to: {output_dir}")

# =============================================================================
# ### CELL 9: FINAL EVALUATION AND CLEANUP
# =============================================================================

# Final evaluation on validation set
if val_dataloader:
    print("\n📊 Final evaluation:")
    final_results = evaluate(model, val_dataloader, id_to_label, device)
    
    if final_results:
        print("\n📈 Classification Report:")
        report = final_results['classification_report']
        for label, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"{label:15} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")

# Memory cleanup
torch.cuda.empty_cache()
print("\n🧹 Memory cleaned up")
print("✅ All done! Ready for inference.") 