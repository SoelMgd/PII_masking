#!/usr/bin/env python3
"""
Dataset verification and visualization for token classification.

This script:
1. Loads processed token classification datasets
2. Visualizes token-label alignments
3. Performs sanity checks on the data
4. Shows statistics and examples for debugging
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetChecker:
    """Checker for token classification datasets."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the checker.
        
        Args:
            data_dir: Directory containing processed datasets
        """
        self.data_dir = Path(data_dir)
        self.train_dataset = None
        self.val_dataset = None
        self.label_mappings = None
        self.stats = None
        
    def load_datasets(self):
        """Load all dataset files."""
        logger.info("Loading datasets...")
        
        # Load training dataset
        train_path = self.data_dir / "train_dataset.pkl"
        if train_path.exists():
            with open(train_path, 'rb') as f:
                self.train_dataset = pickle.load(f)
            logger.info(f"Loaded training dataset: {len(self.train_dataset['texts'])} examples")
        else:
            logger.warning(f"Training dataset not found at {train_path}")
            
        # Load validation dataset
        val_path = self.data_dir / "val_dataset.pkl"
        if val_path.exists():
            with open(val_path, 'rb') as f:
                self.val_dataset = pickle.load(f)
            logger.info(f"Loaded validation dataset: {len(self.val_dataset['texts'])} examples")
        else:
            logger.warning(f"Validation dataset not found at {val_path}")
            
        # Load label mappings
        label_path = self.data_dir / "label_mappings.json"
        if label_path.exists():
            with open(label_path, 'r') as f:
                self.label_mappings = json.load(f)
            logger.info(f"Loaded label mappings: {self.label_mappings['num_labels']} labels")
        else:
            logger.warning(f"Label mappings not found at {label_path}")
            
        # Load statistics
        stats_path = self.data_dir / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            logger.info("Loaded dataset statistics")
        else:
            logger.warning(f"Dataset statistics not found at {stats_path}")
    
    def print_dataset_overview(self):
        """Print overview of the datasets."""
        print("\n" + "="*80)
        print("📊 DATASET OVERVIEW")
        print("="*80)
        
        if self.stats:
            combined_stats = self.stats.get('combined_stats', {})
            print(f"📈 Total examples: {combined_stats.get('total_examples', 'N/A')}")
            print(f"📈 Total tokens: {combined_stats.get('total_tokens', 'N/A')}")
            print(f"📈 Average sequence length: {combined_stats.get('avg_sequence_length', 'N/A'):.1f}")
            print(f"📈 Non-O token ratio: {combined_stats.get('non_o_ratio', 'N/A'):.3f}")
            print(f"📈 Unique labels: {combined_stats.get('unique_labels', 'N/A')}")
            
            print(f"\n🔄 Train/Val split:")
            print(f"   Training: {self.stats.get('train_size', 'N/A')} examples")
            print(f"   Validation: {self.stats.get('val_size', 'N/A')} examples")
            print(f"   Val ratio: {self.stats.get('val_ratio', 'N/A')}")
        
        if self.label_mappings:
            print(f"\n🏷️  Label mappings ({self.label_mappings['num_labels']} labels):")
            for label_id, label_name in self.label_mappings['id_to_label'].items():
                print(f"   {label_id}: {label_name}")
    
    def print_label_distribution(self, dataset_name: str, dataset: Dict[str, Any]):
        """Print label distribution for a dataset."""
        if not dataset:
            return
            
        print(f"\n📊 LABEL DISTRIBUTION - {dataset_name.upper()}")
        print("-" * 60)
        
        # Count labels
        label_counts = defaultdict(int)
        total_tokens = 0
        
        for labels in dataset['labels']:
            for label in labels:
                label_counts[label] += 1
                total_tokens += 1
        
        # Sort by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Label':<20} {'Count':<10} {'Percentage':<10}")
        print("-" * 40)
        for label, count in sorted_labels:
            percentage = (count / total_tokens) * 100
            print(f"{label:<20} {count:<10} {percentage:<10.2f}%")
        
        print(f"\nTotal tokens: {total_tokens}")
    
    def visualize_examples(self, dataset_name: str, dataset: Dict[str, Any], num_examples: int = 3):
        """Visualize examples with token-label alignment."""
        if not dataset:
            return
            
        print(f"\n🔍 EXAMPLE VISUALIZATION - {dataset_name.upper()}")
        print("=" * 80)
        
        # Select random examples
        num_examples = min(num_examples, len(dataset['texts']))
        indices = random.sample(range(len(dataset['texts'])), num_examples)
        
        for i, idx in enumerate(indices):
            text = dataset['texts'][idx]
            tokens = dataset['tokens'][idx]
            labels = dataset['labels'][idx]
            
            print(f"\n📝 Example {i+1} (index {idx}):")
            print(f"Text length: {len(text)} characters, {len(tokens)} tokens")
            print(f"Original text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            # Show token-label alignment
            print(f"\nToken-Label alignment:")
            print("-" * 60)
            
            # Group consecutive tokens with same label for better visualization
            grouped_tokens = []
            current_group = {'tokens': [], 'label': labels[0] if labels else 'O'}
            
            for token, label in zip(tokens, labels):
                if label == current_group['label']:
                    current_group['tokens'].append(token)
                else:
                    grouped_tokens.append(current_group)
                    current_group = {'tokens': [token], 'label': label}
            
            if current_group['tokens']:
                grouped_tokens.append(current_group)
            
            # Display grouped tokens
            for group in grouped_tokens:
                token_text = ''.join(group['tokens'])
                label = group['label']
                
                if label != 'O':
                    print(f"🔴 [{label}]: '{token_text}'")
                else:
                    # Show only first few O tokens to avoid clutter
                    if len(token_text) > 50:
                        token_text = token_text[:50] + '...'
                    print(f"⚪ [O]: '{token_text}'")
            
            # Show individual tokens for detailed inspection
            print(f"\nDetailed token breakdown:")
            for j, (token, label) in enumerate(zip(tokens[:20], labels[:20])):  # Show first 20 tokens
                marker = "🔴" if label != 'O' else "⚪"
                print(f"  {j:2d}: {marker} '{token}' -> {label}")
            
            if len(tokens) > 20:
                print(f"  ... and {len(tokens) - 20} more tokens")
            
            print("-" * 60)
    
    def check_data_consistency(self):
        """Perform consistency checks on the data."""
        print(f"\n🔍 DATA CONSISTENCY CHECKS")
        print("=" * 80)
        
        issues = []
        
        for dataset_name, dataset in [("Training", self.train_dataset), ("Validation", self.val_dataset)]:
            if not dataset:
                continue
                
            print(f"\n📋 Checking {dataset_name} dataset...")
            
            texts = dataset['texts']
            tokens = dataset['tokens']
            labels = dataset['labels']
            label_ids = dataset['label_ids']
            
            # Check lengths match
            if not (len(texts) == len(tokens) == len(labels) == len(label_ids)):
                issues.append(f"{dataset_name}: Mismatched lengths - texts:{len(texts)}, tokens:{len(tokens)}, labels:{len(labels)}, label_ids:{len(label_ids)}")
            
            # Check each example
            for i in range(min(len(texts), 100)):  # Check first 100 examples
                text = texts[i]
                token_list = tokens[i]
                label_list = labels[i]
                label_id_list = label_ids[i]
                
                # Check token-label length match
                if len(token_list) != len(label_list):
                    issues.append(f"{dataset_name} example {i}: Token count ({len(token_list)}) != label count ({len(label_list)})")
                
                # Check label-label_id length match
                if len(label_list) != len(label_id_list):
                    issues.append(f"{dataset_name} example {i}: Label count ({len(label_list)}) != label_id count ({len(label_id_list)})")
                
                # Check if tokens can reconstruct text (approximately)
                reconstructed = ''.join(token_list)
                if reconstructed != text:
                    # This might be expected due to tokenization, but log for inspection
                    if len(issues) < 5:  # Limit to avoid spam
                        issues.append(f"{dataset_name} example {i}: Text reconstruction mismatch (length: {len(text)} vs {len(reconstructed)})")
                
                # Check label consistency with mappings
                if self.label_mappings:
                    for j, (label, label_id) in enumerate(zip(label_list, label_id_list)):
                        expected_id = self.label_mappings['label_to_id'].get(label)
                        if expected_id != label_id:
                            issues.append(f"{dataset_name} example {i}, token {j}: Label '{label}' has ID {label_id}, expected {expected_id}")
        
        # Report issues
        if issues:
            print(f"\n❌ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print(f"\n✅ No consistency issues found!")
    
    def analyze_tokenization_quality(self):
        """Analyze tokenization quality and alignment."""
        print(f"\n🔍 TOKENIZATION QUALITY ANALYSIS")
        print("=" * 80)
        
        if not self.train_dataset:
            print("No training dataset available for analysis")
            return
        
        texts = self.train_dataset['texts']
        tokens = self.train_dataset['tokens']
        labels = self.train_dataset['labels']
        
        # Analyze first few examples in detail
        print(f"\nDetailed tokenization analysis (first 3 examples):")
        
        for i in range(min(3, len(texts))):
            text = texts[i]
            token_list = tokens[i]
            label_list = labels[i]
            
            print(f"\n--- Example {i+1} ---")
            print(f"Original text ({len(text)} chars): '{text}'")
            print(f"Reconstructed ({len(''.join(token_list))} chars): '{''.join(token_list)}'")
            print(f"Match: {text == ''.join(token_list)}")
            
            # Show PII tokens
            pii_tokens = [(j, token, label) for j, (token, label) in enumerate(zip(token_list, label_list)) if label != 'O']
            if pii_tokens:
                print(f"PII tokens found:")
                for j, token, label in pii_tokens:
                    print(f"  Token {j}: '{token}' -> {label}")
            else:
                print(f"No PII tokens in this example")
        
        # Overall statistics
        total_examples = len(texts)
        perfect_matches = sum(1 for i in range(total_examples) if texts[i] == ''.join(tokens[i]))
        
        print(f"\n📊 Overall tokenization statistics:")
        print(f"  Perfect text reconstruction: {perfect_matches}/{total_examples} ({perfect_matches/total_examples*100:.1f}%)")
        
        # Token length distribution
        token_lengths = [len(token_list) for token_list in tokens]
        avg_length = sum(token_lengths) / len(token_lengths)
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        
        print(f"  Sequence lengths - Avg: {avg_length:.1f}, Min: {min_length}, Max: {max_length}")
    
    def check_label_coverage(self):
        """Check if all expected PII types are covered."""
        print(f"\n🔍 LABEL COVERAGE ANALYSIS")
        print("=" * 80)
        
        if not self.label_mappings:
            print("No label mappings available")
            return
        
        # Expected PII types (based on common PII categories)
        expected_pii_types = {
            'PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'CREDIT_CARD', 
            'SSN', 'DATE', 'ORGANIZATION', 'LOCATION', 'URL',
            'IP_ADDRESS', 'IBAN', 'LICENSE_PLATE'
        }
        
        # Get actual labels (excluding 'O')
        actual_labels = set(self.label_mappings['id_to_label'].values()) - {'O'}
        
        print(f"Expected PII types: {sorted(expected_pii_types)}")
        print(f"Actual labels found: {sorted(actual_labels)}")
        
        missing = expected_pii_types - actual_labels
        extra = actual_labels - expected_pii_types
        
        if missing:
            print(f"\n⚠️  Missing expected PII types: {sorted(missing)}")
        
        if extra:
            print(f"\n➕ Additional labels found: {sorted(extra)}")
        
        if not missing and not extra:
            print(f"\n✅ Perfect label coverage!")

def main():
    """Main function for dataset checking."""
    parser = argparse.ArgumentParser(description="Check and visualize token classification datasets")
    
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to visualize (default: 3)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis including tokenization quality"
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = DatasetChecker(data_dir=Path(args.data_dir))
    
    # Load datasets
    checker.load_datasets()
    
    if not checker.train_dataset and not checker.val_dataset:
        logger.error("No datasets found. Make sure to run dataset_processing.py first.")
        return
    
    # Print overview
    checker.print_dataset_overview()
    
    # Print label distributions
    if checker.train_dataset:
        checker.print_label_distribution("Training", checker.train_dataset)
    
    if checker.val_dataset:
        checker.print_label_distribution("Validation", checker.val_dataset)
    
    # Visualize examples
    if checker.train_dataset:
        checker.visualize_examples("Training", checker.train_dataset, args.num_examples)
    
    if checker.val_dataset:
        checker.visualize_examples("Validation", checker.val_dataset, min(2, args.num_examples))
    
    # Perform consistency checks
    checker.check_data_consistency()
    
    # Check label coverage
    checker.check_label_coverage()
    
    # Detailed analysis if requested
    if args.detailed:
        checker.analyze_tokenization_quality()
    
    print(f"\n🎉 Dataset checking completed!")
    print(f"📁 Data directory: {args.data_dir}")

if __name__ == "__main__":
    main() 