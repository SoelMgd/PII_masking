#!/usr/bin/env python3
"""
Debug script to analyze BERT token-to-entity conversion and compare with ground truth.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pii_masking.text_processing import EntitySpan
from pii_masking.data_loader import PIIExample

# Import functions from eval.py
from eval import (
    load_processed_data, 
    split_data, 
    convert_bert_example_to_pii_example,
    BERTTokenClassificationModel
)

def analyze_example(bert_example: Dict, model: BERTTokenClassificationModel, example_idx: int):
    """Analyze a single example in detail."""
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {example_idx}")
    print(f"{'='*80}")
    
    # Original text and tokens
    original_text = bert_example['original_text']
    tokens = bert_example['tokens']
    labels = bert_example['labels']
    token_spans = bert_example['token_spans']
    
    print(f"Original text: {repr(original_text)}")
    print(f"Length: {len(original_text)}")
    
    # Show token-level information
    print(f"\nToken-level analysis:")
    print(f"{'Token':<15} {'Label':<15} {'Span':<10} {'Text'}")
    print(f"{'-'*60}")
    
    for i, (token, label, (start, end)) in enumerate(zip(tokens, labels, token_spans)):
        if token not in ['[CLS]', '[SEP]', '[PAD]'] and label != 'O':
            token_text = original_text[start:end] if start < len(original_text) and end <= len(original_text) else "OUT_OF_BOUNDS"
            print(f"{token:<15} {label:<15} {start}-{end:<5} {repr(token_text)}")
    
    # Convert to PIIExample format (ground truth)
    pii_example = convert_bert_example_to_pii_example(bert_example)
    
    print(f"\nGround truth entities (from token conversion):")
    for span_data in pii_example.span_labels:
        start, end, label = span_data[0], span_data[1], span_data[2]
        entity_text = original_text[start:end]
        print(f"  {label}: {start}-{end} = {repr(entity_text)}")
    
    # Get model predictions
    prediction = model.predict_single(original_text)
    
    print(f"\nModel predictions:")
    for span in prediction.spans:
        print(f"  {span.entity_type}: {span.start}-{span.end} = {repr(span.text)}")
    
    # Compare
    print(f"\nComparison:")
    true_set = {(span[0], span[1], span[2]) for span in pii_example.span_labels}
    pred_set = {(span.start, span.end, span.entity_type) for span in prediction.spans}
    
    print(f"True spans: {true_set}")
    print(f"Pred spans: {pred_set}")
    print(f"Matches: {true_set & pred_set}")
    print(f"Missed (FN): {true_set - pred_set}")
    print(f"False alarms (FP): {pred_set - true_set}")
    
    return len(true_set & pred_set), len(pred_set - true_set), len(true_set - pred_set)

def main():
    """Debug the evaluation process."""
    
    print("BERT Token Classification - Debug Evaluation")
    print("="*60)
    
    # Paths
    model_path = "/Users/twin/Documents/pii-masking-200k/models/bert_token_classif"
    data_path = "/Users/twin/Documents/pii-masking-200k/data/processed_data_bert"
    
    # Load data
    print("Loading data...")
    examples, label2id, id2label = load_processed_data(data_path)
    train_examples, val_examples, test_examples = split_data(
        examples, train_split=0.8, val_split=0.1, test_split=0.1
    )
    
    # Initialize model
    print("Loading model...")
    model = BERTTokenClassificationModel(model_path)
    if not model.initialize():
        print("Failed to initialize model")
        return
    
    # Analyze first few examples
    print(f"\nAnalyzing first 5 test examples...")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i in range(min(5, len(test_examples))):
        tp, fp, fn = analyze_example(test_examples[i], model, i+1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY (first 5 examples)")
    print(f"{'='*80}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"Precision: {precision:.4f}")
    
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Recall: {recall:.4f}")
    
    if total_tp + total_fp > 0 and total_tp + total_fn > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main() 