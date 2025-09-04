"""
Custom evaluator for efficient PII masking using JSON output and regex matching.

This evaluator:
1. Parses JSON predictions from the model
2. Uses regex matching to find entity positions in original text
3. Converts matches to spans (start, end, label)
4. Compares predicted spans with ground truth spans using exact matching
5. Calculates precision, recall, F1-score per class and globally
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from pii_masking.data_loader import PIIExample
from pii_masking.text_processing import EntitySpan, PIIPrediction

logger = logging.getLogger(__name__)

@dataclass
class CustomEvaluationResult:
    """Results from custom PII evaluation."""
    precision: float
    recall: float
    f1_score: float
    per_class_metrics: Dict[str, Dict[str, float]]
    total_examples: int
    successful_predictions: int
    total_true_entities: int
    total_pred_entities: int
    total_correct_entities: int
    experiment_name: str
    model_name: str
    config: Dict
    detailed_results: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'per_class_metrics': {
                k: {
                    'precision': float(v['precision']),
                    'recall': float(v['recall']),
                    'f1_score': float(v['f1_score']),
                    'support': int(v['support']),
                    'true_positives': int(v['true_positives']),
                    'false_positives': int(v['false_positives']),
                    'false_negatives': int(v['false_negatives'])
                } for k, v in self.per_class_metrics.items()
            },
            'total_examples': int(self.total_examples),
            'successful_predictions': int(self.successful_predictions),
            'total_true_entities': int(self.total_true_entities),
            'total_pred_entities': int(self.total_pred_entities),
            'total_correct_entities': int(self.total_correct_entities),
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'config': self.config,
            'detailed_results': self.detailed_results or []
        }

class CustomPIIEvaluator:
    """Custom evaluator for PII masking using JSON output and exact span matching."""
    
    def __init__(self):
        pass
    
    def extract_true_spans(self, example: PIIExample) -> List[EntitySpan]:
        """
        Extract true entity spans from ground truth.
        
        Args:
            example: PIIExample with span_labels
            
        Returns:
            List of EntitySpan objects (excluding "O" labels)
        """
        spans = []
        
        for span_data in example.span_labels:
            if len(span_data) >= 3:
                start, end, label = span_data[0], span_data[1], span_data[2]
                
                entity_type = label.replace('B-', '').replace('I-', '')
                if '_' in entity_type:
                    entity_type = entity_type.split('_')[0]
                
                if entity_type != 'O':
                    text = example.unmasked_text[start:end] if start < len(example.unmasked_text) else ""
                    spans.append(EntitySpan(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        text=text
                    ))
        
        return spans
    
    def exact_match_spans(self, true_spans: List[EntitySpan], 
                         pred_spans: List[EntitySpan]) -> Tuple[int, int, int]:
        """
        Calculate exact matches between true and predicted spans.
        
        Args:
            true_spans: Ground truth spans
            pred_spans: Predicted spans
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        true_set = {(span.start, span.end, span.entity_type) for span in true_spans}
        pred_set = {(span.start, span.end, span.entity_type) for span in pred_spans}
        
        true_positives = len(true_set.intersection(pred_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)
        
        return true_positives, false_positives, false_negatives
    
    def calculate_metrics(self, true_positives: int, false_positives: int, 
                         false_negatives: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_single(self, example: PIIExample, prediction: PIIPrediction, 
                       save_details: bool = False) -> Dict[str, Any]:
        """Evaluate a single example."""
        true_spans = self.extract_true_spans(example)
        pred_spans = prediction.spans
        
        tp, fp, fn = self.exact_match_spans(true_spans, pred_spans)
        
        per_class_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        true_by_class = defaultdict(set)
        pred_by_class = defaultdict(set)
        
        for span in true_spans:
            true_by_class[span.entity_type].add((span.start, span.end))
        
        for span in pred_spans:
            pred_by_class[span.entity_type].add((span.start, span.end))
        
        all_classes = set(true_by_class.keys()) | set(pred_by_class.keys())
        
        for entity_type in all_classes:
            true_positions = true_by_class[entity_type]
            pred_positions = pred_by_class[entity_type]
            
            class_tp = len(true_positions.intersection(pred_positions))
            class_fp = len(pred_positions - true_positions)
            class_fn = len(true_positions - pred_positions)
            
            per_class_counts[entity_type]['tp'] = class_tp
            per_class_counts[entity_type]['fp'] = class_fp
            per_class_counts[entity_type]['fn'] = class_fn
        
        result = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'per_class_counts': dict(per_class_counts),
            'num_true_entities': len(true_spans),
            'num_pred_entities': len(pred_spans)
        }
        
        if save_details:
            result.update({
                'example_text': example.unmasked_text[:200] + '...' if len(example.unmasked_text) > 200 else example.unmasked_text,
                'true_spans': [(s.start, s.end, s.entity_type, s.text) for s in true_spans],
                'pred_spans': [(s.start, s.end, s.entity_type, s.text) for s in pred_spans],
                'prediction_entities': prediction.entities
            })
        
        return result
    
    def evaluate_dataset(self, examples: List[PIIExample], 
                        predictions: List[PIIPrediction],
                        experiment_name: str = "evaluation",
                        model_name: str = "unknown",
                        config: Dict = None,
                        save_details: bool = False) -> CustomEvaluationResult:
        """
        Evaluate predictions against ground truth examples.
        
        Args:
            examples: List of PIIExample objects
            predictions: List of PIIPrediction objects
            experiment_name: Name for this evaluation run
            model_name: Name of the model being evaluated
            config: Configuration used for the experiment
            save_details: Whether to save detailed results for each example
            
        Returns:
            CustomEvaluationResult with metrics
        """
        if len(examples) != len(predictions):
            raise ValueError(f"Number of examples ({len(examples)}) must match predictions ({len(predictions)})")
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        per_class_totals = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        detailed_results = [] if save_details else None
        successful_predictions = 0
        
        for example, prediction in zip(examples, predictions):
            try:
                result = self.evaluate_single(example, prediction, save_details)
                
                total_tp += result['true_positives']
                total_fp += result['false_positives']
                total_fn += result['false_negatives']
                
                for entity_type, counts in result['per_class_counts'].items():
                    per_class_totals[entity_type]['tp'] += counts['tp']
                    per_class_totals[entity_type]['fp'] += counts['fp']
                    per_class_totals[entity_type]['fn'] += counts['fn']
                
                successful_predictions += 1
                
                if save_details:
                    detailed_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                continue
        
        global_metrics = self.calculate_metrics(total_tp, total_fp, total_fn)
        
        per_class_metrics = {}
        for entity_type, counts in per_class_totals.items():
            metrics = self.calculate_metrics(counts['tp'], counts['fp'], counts['fn'])
            metrics['support'] = counts['tp'] + counts['fn']
            per_class_metrics[entity_type] = metrics
        
        return CustomEvaluationResult(
            precision=global_metrics['precision'],
            recall=global_metrics['recall'],
            f1_score=global_metrics['f1_score'],
            per_class_metrics=per_class_metrics,
            total_examples=len(examples),
            successful_predictions=successful_predictions,
            total_true_entities=total_tp + total_fn,
            total_pred_entities=total_tp + total_fp,
            total_correct_entities=total_tp,
            experiment_name=experiment_name,
            model_name=model_name,
            config=config or {},
            detailed_results=detailed_results
        )
    
    def print_results(self, result: CustomEvaluationResult, top_k: int = 10):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"PII MASKING EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Experiment: {result.experiment_name}")
        print(f"Model: {result.model_name}")
        print(f"Examples: {result.total_examples} (successful: {result.successful_predictions})")
        print()
        
        print("GLOBAL METRICS:")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1-Score:  {result.f1_score:.4f}")
        print()
        
        print("ENTITY STATISTICS:")
        print(f"  True entities:      {result.total_true_entities}")
        print(f"  Predicted entities: {result.total_pred_entities}")
        print(f"  Correct entities:   {result.total_correct_entities}")
        print()
        
        print("PER-CLASS METRICS (Top {}):".format(top_k))
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        sorted_classes = sorted(result.per_class_metrics.items(), 
                              key=lambda x: x[1]['f1_score'], reverse=True)
        
        for entity_type, metrics in sorted_classes[:top_k]:
            print(f"{entity_type:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}")
        
        if len(sorted_classes) > top_k:
            print(f"... and {len(sorted_classes) - top_k} more classes")
        
        print(f"{'='*60}")
    
    def save_results(self, result: CustomEvaluationResult, output_file: str):
        """Save evaluation results to JSON file."""
        import json
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")

def evaluate_predictions(examples: List[PIIExample], 
                        predictions: List[PIIPrediction],
                        experiment_name: str = "evaluation",
                        model_name: str = "unknown",
                        config: Dict = None,
                        output_file: str = None,
                        save_details: bool = False,
                        print_results: bool = True) -> CustomEvaluationResult:
    """
    Convenience function to evaluate predictions and optionally save results.
    
    Args:
        examples: Ground truth examples
        predictions: Model predictions
        experiment_name: Name for this evaluation
        model_name: Model identifier
        config: Experiment configuration
        output_file: Optional file to save results
        save_details: Whether to save detailed per-example results
        print_results: Whether to print results to console
        
    Returns:
        CustomEvaluationResult object
    """
    evaluator = CustomPIIEvaluator()
    
    result = evaluator.evaluate_dataset(
        examples=examples,
        predictions=predictions,
        experiment_name=experiment_name,
        model_name=model_name,
        config=config,
        save_details=save_details
    )
    
    if print_results:
        evaluator.print_results(result)
    
    if output_file:
        evaluator.save_results(result, output_file)
    
    return result 