#!/usr/bin/env python3
"""
Evaluation script for Mistral token classification model.

This script:
1. Loads the trained token classification model
2. Evaluates on test datasets
3. Calculates detailed metrics
4. Provides error analysis and examples

Designed to run on Kaggle after training.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralTokenClassifier(nn.Module):
    """Mistral model with token classification head (same as training)."""
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for inference."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        return logits

class PIITokenDataset:
    """Dataset class for evaluation (simplified version)."""
    
    def __init__(self, dataset: Dict[str, Any], tokenizer, max_length: int = 512):
        self.texts = dataset['texts']
        self.tokens = dataset['tokens']
        self.labels = dataset['labels']
        self.label_ids = dataset['label_ids']
        self.label_to_id = dataset['label_to_id']
        self.id_to_label = dataset['id_to_label']
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        original_tokens = self.tokens[idx]
        original_labels = self.label_ids[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        aligned_labels = self._align_labels_with_tokenization(
            text, original_tokens, original_labels, encoding
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'original_text': text,
            'original_tokens': original_tokens,
            'original_labels': [self.id_to_label[lid] for lid in original_labels]
        }
    
    def _align_labels_with_tokenization(self, text: str, original_tokens: List[str], 
                                       original_labels: List[int], encoding) -> List[int]:
        """Align original token labels with model tokenization."""
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        
        char_labels = [0] * len(text)
        current_pos = 0
        
        for token, label_id in zip(original_tokens, original_labels):
            token_start = text.find(token, current_pos)
            if token_start != -1:
                token_end = token_start + len(token)
                for i in range(token_start, min(token_end, len(text))):
                    char_labels[i] = label_id
                current_pos = token_end
        
        aligned_labels = []
        for start, end in offset_mapping:
            if start == end:
                aligned_labels.append(-100)
            else:
                span_labels = char_labels[start:end]
                if span_labels:
                    label = next((l for l in span_labels if l != 0), 0)
                    aligned_labels.append(label)
                else:
                    aligned_labels.append(0)
        
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(-100)
        aligned_labels = aligned_labels[:self.max_length]
        
        return aligned_labels

class TokenClassificationEvaluator:
    """Evaluator for token classification model."""
    
    def __init__(self, model_dir: str):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing the trained model
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration
        with open(self.model_dir / "pytorch_model.bin", 'rb') as f:
            self.model_state = torch.load(f, map_location=self.device)
        
        self.config = self.model_state['config']
        self.label_to_id = self.model_state['label_to_id']
        self.id_to_label = self.model_state['id_to_label']
        self.num_labels = self.model_state['num_labels']
        self.model_name = self.model_state['model_name']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Initialize and load model
        self.model = MistralTokenClassifier(self.model_name, self.num_labels)
        self.model.classifier.load_state_dict(self.model_state['classifier_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_dir}")
        logger.info(f"Number of labels: {self.num_labels}")
    
    def predict_batch(self, dataloader: DataLoader) -> Tuple[List[int], List[int], List[Dict]]:
        """
        Make predictions on a batch of data.
        
        Returns:
            Tuple of (all_predictions, all_labels, detailed_results)
        """
        all_predictions = []
        all_labels = []
        detailed_results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move inputs to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get predictions
                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                
                # Process each example in the batch
                for i in range(len(input_ids)):
                    example_labels = labels[i].cpu().numpy()
                    example_predictions = predictions[i].cpu().numpy()
                    
                    # Filter out ignored labels (-100)
                    mask = example_labels != -100
                    valid_labels = example_labels[mask]
                    valid_predictions = example_predictions[mask]
                    
                    all_labels.extend(valid_labels)
                    all_predictions.extend(valid_predictions)
                    
                    # Store detailed results
                    detailed_results.append({
                        'original_text': batch['original_text'][i],
                        'original_tokens': batch['original_tokens'][i],
                        'original_labels': batch['original_labels'][i],
                        'predicted_labels': [self.id_to_label[pid] for pid in valid_predictions],
                        'true_labels': [self.id_to_label[lid] for lid in valid_labels]
                    })
        
        return all_predictions, all_labels, detailed_results
    
    def calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        
        # Overall metrics
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_micro = f1_score(labels, predictions, average='micro')
        
        # Per-class metrics
        target_names = [self.id_to_label[i] for i in range(self.num_labels)]
        class_report = classification_report(
            labels, 
            predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'label_names': target_names
        }
    
    def analyze_errors(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Analyze prediction errors."""
        error_analysis = {
            'total_examples': len(detailed_results),
            'perfect_matches': 0,
            'label_errors': defaultdict(int),
            'common_mistakes': defaultdict(int),
            'examples_with_errors': []
        }
        
        for result in detailed_results:
            true_labels = result['true_labels']
            pred_labels = result['predicted_labels']
            
            # Check if perfect match
            if true_labels == pred_labels:
                error_analysis['perfect_matches'] += 1
            else:
                # Analyze errors
                error_analysis['examples_with_errors'].append(result)
                
                for true_label, pred_label in zip(true_labels, pred_labels):
                    if true_label != pred_label:
                        error_analysis['label_errors'][true_label] += 1
                        error_analysis['common_mistakes'][f"{true_label} -> {pred_label}"] += 1
        
        # Calculate accuracy
        error_analysis['accuracy'] = error_analysis['perfect_matches'] / error_analysis['total_examples']
        
        return error_analysis
    
    def evaluate_dataset(self, dataset_path: str, batch_size: int = 16) -> Dict[str, Any]:
        """
        Evaluate on a complete dataset.
        
        Args:
            dataset_path: Path to the dataset file
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        dataset = PIITokenDataset(dataset_dict, self.tokenizer, self.config['max_length'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Evaluating on {len(dataset)} examples")
        
        # Make predictions
        predictions, labels, detailed_results = self.predict_batch(dataloader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels)
        
        # Error analysis
        error_analysis = self.analyze_errors(detailed_results)
        
        return {
            'metrics': metrics,
            'error_analysis': error_analysis,
            'detailed_results': detailed_results[:100]  # Keep only first 100 for storage
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print a comprehensive evaluation report."""
        metrics = results['metrics']
        error_analysis = results['error_analysis']
        
        print(f"\n{'='*80}")
        print(f"🎯 TOKEN CLASSIFICATION EVALUATION REPORT")
        print(f"{'='*80}")
        
        # Overall metrics
        print(f"\n📊 OVERALL METRICS:")
        print(f"   F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"   F1-Score (Macro):    {metrics['f1_macro']:.4f}")
        print(f"   F1-Score (Micro):    {metrics['f1_micro']:.4f}")
        print(f"   Accuracy:            {error_analysis['accuracy']:.4f}")
        
        # Per-class metrics (top 10 by support)
        print(f"\n📋 TOP CLASSES BY SUPPORT:")
        class_report = metrics['classification_report']
        
        # Sort classes by support
        classes_by_support = []
        for label, stats in class_report.items():
            if isinstance(stats, dict) and 'support' in stats:
                classes_by_support.append((label, stats))
        
        classes_by_support.sort(key=lambda x: x[1]['support'], reverse=True)
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for label, stats in classes_by_support[:10]:
            print(f"{label:<15} {stats['precision']:<10.3f} {stats['recall']:<10.3f} "
                  f"{stats['f1-score']:<10.3f} {stats['support']:<10}")
        
        # Error analysis
        print(f"\n🔍 ERROR ANALYSIS:")
        print(f"   Total Examples:      {error_analysis['total_examples']}")
        print(f"   Perfect Matches:     {error_analysis['perfect_matches']}")
        print(f"   Examples with Errors: {len(error_analysis['examples_with_errors'])}")
        
        # Most common mistakes
        print(f"\n❌ MOST COMMON MISTAKES:")
        common_mistakes = sorted(
            error_analysis['common_mistakes'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for mistake, count in common_mistakes[:10]:
            print(f"   {mistake}: {count}")
        
        print(f"\n{'='*80}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Mistral token classification model")
    
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model")
    parser.add_argument("--test-dataset", required=True, help="Path to test dataset")
    parser.add_argument("--output-file", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = TokenClassificationEvaluator(args.model_dir)
        
        # Evaluate
        results = evaluator.evaluate_dataset(args.test_dataset, args.batch_size)
        
        # Print report
        evaluator.print_evaluation_report(results)
        
        # Save results
        evaluator.save_results(results, args.output_file)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_file}")
        print(f"📊 F1-Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()