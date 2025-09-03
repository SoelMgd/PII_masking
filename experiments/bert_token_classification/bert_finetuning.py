#!/usr/bin/env python3
"""
BERT Fine-tuning for PII Token Classification

This script fine-tunes a DistilBERT model for PII token classification using our processed dataset.
The script is structured with cell delimiters for easy conversion to Kaggle notebook.

Key features:
- Uses our custom PIIPrediction objects for evaluation
- Compatible with our existing evaluation framework
- Optimized for Kaggle GPU training
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd

# Disable wandb logging completely
os.environ["WANDB_DISABLED"] = "true"

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Transformers and ML libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from datasets import Dataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set up comprehensive logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Force stdout for notebooks
    ],
    force=True  # Override any existing logging config
)
logger = logging.getLogger(__name__)

# Enable transformers logging
import transformers
transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

# Enable progress bars
from tqdm.auto import tqdm
tqdm.pandas()

# Force flush for real-time display
sys.stdout.flush()
sys.stderr.flush()

# Check if we're in Kaggle environment
IN_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ
if IN_KAGGLE:
    # Kaggle paths
    DATA_PATH = "/kaggle/input/pii-dataset/processed_data"
    OUTPUT_PATH = "/kaggle/working"
    DATASET_PATH = "/kaggle/input/pii-dataset/data"
    
    # Kaggle-specific optimizations
    print("🔧 KAGGLE ENVIRONMENT DETECTED")
    print("   - Optimizing for notebook display")
    print("   - Using enhanced progress tracking")
    
    # Force display settings for Kaggle
    from IPython.display import clear_output
    import time
    
else:
    # Local paths
    DATA_PATH = "processed_data"
    OUTPUT_PATH = "outputs"
    DATASET_PATH = "../../data"

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
print(f"Data path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# ============================================================================
# CELL 2: Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model configuration
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None  # Set to number for testing
    
    # Output parameters
    save_model: bool = True
    evaluate_on_test: bool = True

config = TrainingConfig()

# Print configuration
print("Training Configuration:")
for key, value in config.__dict__.items():
    print(f"  {key}: {value}")

# ============================================================================
# CELL 3: Data Loading and Preprocessing
# ============================================================================

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
    logger.info(f"Labels: {sorted(label2id.keys())}")
    
    return examples, label2id, id2label

# Load data
examples, label2id, id2label = load_processed_data(DATA_PATH)

# Apply max_samples limit if specified
if config.max_samples:
    examples = examples[:config.max_samples]
    logger.info(f"Limited to {len(examples)} examples for testing")

print(f"\nDataset Statistics:")
print(f"Total examples: {len(examples)}")
print(f"Number of unique labels: {len(label2id)}")

# ============================================================================
# CELL 4: Data Splitting and Dataset Creation
# ============================================================================

def split_data(examples: List[Dict], train_split: float, val_split: float, test_split: float):
    """Split data into train/val/test sets."""
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    n_total = len(examples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    logger.info(f"Data split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    
    return train_examples, val_examples, test_examples

def create_dataset(examples: List[Dict], tokenizer, label2id: Dict[str, int], max_length: int):
    """
    Create HuggingFace Dataset from processed examples.
    """
    
    def tokenize_and_align_labels(examples_batch):
        """Tokenize and align labels for a batch of examples."""
        
        tokenized_inputs = tokenizer(
            [ex['tokens'] for ex in examples_batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        labels = []
        for i, example in enumerate(examples_batch):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 (ignored in loss)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First token of a word gets the label
                    if word_idx < len(example['labels']):
                        label = example['labels'][word_idx]
                        label_ids.append(label2id.get(label, label2id['O']))
                    else:
                        label_ids.append(label2id['O'])
                else:
                    # Subsequent tokens of the same word get -100 (ignored)
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Convert to HuggingFace Dataset format
    dataset_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        tokenized = tokenize_and_align_labels(batch)
        
        dataset_dict['input_ids'].extend(tokenized['input_ids'].tolist())
        dataset_dict['attention_mask'].extend(tokenized['attention_mask'].tolist())
        dataset_dict['labels'].extend(tokenized['labels'])
    
    return Dataset.from_dict(dataset_dict)

# Split data
train_examples, val_examples, test_examples = split_data(
    examples, config.train_split, config.val_split, config.test_split
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Create datasets
print("Creating datasets...")
train_dataset = create_dataset(train_examples, tokenizer, label2id, config.max_length)
val_dataset = create_dataset(val_examples, tokenizer, label2id, config.max_length)
test_dataset = create_dataset(test_examples, tokenizer, label2id, config.max_length)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# ============================================================================
# CELL 5: Model Initialization
# ============================================================================

def initialize_model(model_name: str, num_labels: int, label2id: Dict[str, int], id2label: Dict[int, str]):
    """Initialize the token classification model."""
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    logger.info(f"Initialized model: {model_name}")
    logger.info(f"Number of parameters: {model.num_parameters():,}")
    
    return model

# Initialize model
model = initialize_model(
    config.model_name,
    len(label2id),
    label2id,
    id2label
)

print(f"Model initialized with {len(label2id)} labels")
print(f"Model parameters: {model.num_parameters():,}")

# ============================================================================
# CELL 6: Training Setup and Custom Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten for sklearn metrics
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    
    # Calculate metrics
    report = classification_report(
        flat_true_labels, 
        flat_predictions, 
        output_dict=True,
        zero_division=0
    )
    
    # Extract key metrics
    metrics = {
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'accuracy': report['accuracy']
    }
    
    return metrics

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_PATH}/bert_pii_model",
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,
    logging_dir=f"{OUTPUT_PATH}/logs",
    logging_steps=50,  # More frequent logging
    logging_first_step=True,  # Log the first step
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,  # Must match eval_steps for load_best_model_at_end
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to=[],  # Disable wandb/tensorboard completely
    dataloader_pin_memory=False,  # Helps with memory on some systems
    disable_tqdm=not IN_KAGGLE,  # Disable native tqdm in Kaggle, enable locally
    log_level="info",  # Ensure info level logging
    logging_nan_inf_filter=False,  # Show all logs
    log_on_each_node=False,  # Avoid duplicate logs
)

print("Training arguments configured:")
print(f"  Epochs: {config.num_epochs}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Output directory: {training_args.output_dir}")

# ============================================================================
# CELL 7: Custom Training Callback for Progress Logging
# ============================================================================

class ProgressCallback(TrainerCallback):
    """Custom callback to display training progress."""
    
    def __init__(self):
        self.training_start_time = None
        self.last_log_time = None
        self.in_kaggle = 'KAGGLE_WORKING_DIR' in os.environ
        
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.training_start_time = time.time()
        print("=" * 80)
        print("🚀 TRAINING STARTED!")
        print("=" * 80)
        logger.info(f"📊 Total steps: {state.max_steps}")
        logger.info(f"📈 Total epochs: {args.num_train_epochs}")
        logger.info(f"🔢 Batch size: {args.per_device_train_batch_size}")
        print("=" * 80)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch) + 1
        print(f"\n🔄 EPOCH {epoch_num}/{args.num_train_epochs} STARTED")
        print("-" * 60)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            max_steps = state.max_steps
            progress = (step / max_steps) * 100 if max_steps > 0 else 0
            
            # Create a simple progress bar
            bar_length = 40
            filled_length = int(bar_length * step // max_steps) if max_steps > 0 else 0
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Enhanced display for Kaggle
            if self.in_kaggle and step % (args.logging_steps * 2) == 0:
                # Clear previous output every few steps in Kaggle for cleaner display
                try:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    print("🎯 BERT PII Training Progress")
                    print("=" * 50)
                except:
                    pass
            
            print(f"\n📊 Step {step:4d}/{max_steps} [{bar}] {progress:.1f}%")
            
            # Training logs
            if "loss" in logs:
                print(f"   📉 Loss: {logs['loss']:.4f}")
            if "learning_rate" in logs:
                print(f"   📚 LR: {logs['learning_rate']:.2e}")
                
            # Evaluation logs
            if "eval_loss" in logs:
                print(f"   🎯 Eval Loss: {logs['eval_loss']:.4f}")
            if "eval_f1" in logs:
                print(f"   🏆 F1 Score: {logs['eval_f1']:.4f}")
            if "eval_precision" in logs:
                print(f"   🎯 Precision: {logs['eval_precision']:.4f}")
            if "eval_recall" in logs:
                print(f"   🔍 Recall: {logs['eval_recall']:.4f}")
            
            # Kaggle-specific: Add timestamp for long training
            if self.in_kaggle:
                import time
                current_time = time.strftime("%H:%M:%S")
                print(f"   ⏰ Time: {current_time}")
                
            # Force flush for immediate display
            import sys
            sys.stdout.flush()
                
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\n📊 Evaluation completed at step {state.global_step}")
        
    def on_save(self, args, state, control, **kwargs):
        print(f"💾 Model checkpoint saved at step {state.global_step}")
        
    def on_train_end(self, args, state, control, **kwargs):
        import time
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETED!")
            print(f"⏰ Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print("=" * 80)

# ============================================================================
# CELL 8: Initialize Trainer and Start Training
# ============================================================================

# Initialize trainer with progress callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,  # Use new parameter name to avoid warning
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[ProgressCallback()],  # Add our custom progress callback
)

print("Trainer initialized. Starting training...")

# Test display capabilities
print("\n🧪 Testing display capabilities...")
print("   ✅ Basic print works")
logger.info("   ✅ Logger works")

if IN_KAGGLE:
    try:
        from IPython.display import clear_output, display
        print("   ✅ IPython display available")
    except ImportError:
        print("   ⚠️  IPython display not available")

# Test progress bar display
print("   📊 Progress bar test:")
test_bar = '█' * 10 + '░' * 30
print(f"      [{test_bar}] 25.0%")

print("\n🎯 Training configuration:")
print(f"   📊 Dataset size: {len(train_dataset)} train, {len(val_dataset)} validation")
print(f"   🔢 Batch size: {config.batch_size}")
print(f"   📈 Learning rate: {config.learning_rate}")
print(f"   ⏰ Epochs: {config.num_epochs}")
print(f"   💾 Logging every {training_args.logging_steps} steps")
print(f"   📊 Evaluation every {training_args.eval_steps} steps")
print(f"   🖥️  Environment: {'Kaggle Notebook' if IN_KAGGLE else 'Local'}")
print(f"   📈 Native tqdm: {'Disabled' if training_args.disable_tqdm else 'Enabled'}")

# Start training with progress tracking
try:
    training_result = trainer.train()
    logger.info("🎉 Training completed successfully!")
except Exception as e:
    logger.error(f"❌ Training failed: {e}")
    raise

print("Training completed!")
print(f"Training loss: {training_result.training_loss:.4f}")

# Save the model if requested
if config.save_model:
    trainer.save_model(f"{OUTPUT_PATH}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_PATH}/final_model")
    print(f"Model saved to {OUTPUT_PATH}/final_model")

# ============================================================================
# CELL 8: Evaluation on Test Set
# ============================================================================

if config.evaluate_on_test:
    print("Evaluating on test set...")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    print("Test Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            print(f"  {key[5:]}: {value:.4f}")
    
    # Save training results
    results_dict = {
        'model_name': config.model_name,
        'training_config': config.__dict__,
        'training_loss': training_result.training_loss,
        'test_results': test_results,
        'label_mappings': {
            'label2id': label2id,
            'id2label': {int(k): v for k, v in id2label.items()},
            'num_labels': len(label2id)
        }
    }
    
    # Save results
    with open(f"{OUTPUT_PATH}/training_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

print(f"\n🎉 Training completed successfully!")
print(f"📊 Results saved to {OUTPUT_PATH}/training_results.json")
print(f"💾 Model saved to {OUTPUT_PATH}/final_model")
print(f"\n✅ Ready for inference testing with bert_test.py!") 