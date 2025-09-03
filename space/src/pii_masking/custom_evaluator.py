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
    """
    Custom evaluator for PII masking using JSON output and exact span matching.
    """
    
    def __init__(self):
        """Initialize the custom evaluator."""
        pass
    
    # Note: parse_json_prediction and json_to_spans methods are no longer needed
    # since we now work directly with PIIPrediction objects that have pre-computed spans
    
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
                
                # Remove BIO prefixes and suffixes for consistency
                entity_type = label.replace('B-', '').replace('I-', '')
                if '_' in entity_type:
                    entity_type = entity_type.split('_')[0]
                
                # Skip "O" labels (non-PII text)
                if entity_type == 'O':
                    continue
                
                # Extract text from original
                text = example.unmasked_text[start:end]
                
                span = EntitySpan(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    text=text
                )
                spans.append(span)
        
        return spans
    
    def match_spans_exact(self, true_spans: List[EntitySpan], pred_spans: List[EntitySpan]) -> Tuple[int, int, int]:
        """
        Match predicted spans with true spans using exact matching.
        
        Args:
            true_spans: Ground truth entity spans
            pred_spans: Predicted entity spans
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Convert true spans to set of (start, end, entity_type) for exact matching
        true_set = {(span.start, span.end, span.entity_type) for span in true_spans}
        pred_set = {(span.start, span.end, span.entity_type) for span in pred_spans}
        
        # Calculate metrics
        true_positives = len(true_set & pred_set)  # Intersection
        false_positives = len(pred_set - true_set)  # Predicted but not true
        false_negatives = len(true_set - pred_set)  # True but not predicted
        
        return true_positives, false_positives, false_negatives
    
    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score from counts.
        
        Args:
            tp: True positives
            fp: False positives  
            fn: False negatives
            
        Returns:
            Dictionary with precision, recall, f1_score
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': tp + fn,  # Total true entities
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    # Note: reconstruct_masked_text method is no longer needed
    # since PIIPrediction objects have pre-computed masked_text
    
    def evaluate_single_example(self, example: PIIExample, prediction: PIIPrediction) -> Dict[str, Any]:
        """
        Evaluate a single example using exact span matching.
        
        Args:
            example: Ground truth PIIExample
            prediction: PIIPrediction object with pre-computed spans
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Extract spans (already computed in PIIPrediction)
            true_spans = self.extract_true_spans(example)
            pred_spans = prediction.spans
            
            # Calculate per-class metrics
            per_class_metrics = {}
            
            # Group spans by entity type
            true_by_type = defaultdict(list)
            pred_by_type = defaultdict(list)
            
            for span in true_spans:
                true_by_type[span.entity_type].append(span)
            
            for span in pred_spans:
                pred_by_type[span.entity_type].append(span)
            
            # Get all entity types
            all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
            
            for entity_type in all_types:
                true_type_spans = true_by_type[entity_type]
                pred_type_spans = pred_by_type[entity_type]
                
                tp, fp, fn = self.match_spans_exact(true_type_spans, pred_type_spans)
                per_class_metrics[entity_type] = self.calculate_metrics(tp, fp, fn)
            
            # Calculate overall metrics
            total_tp, total_fp, total_fn = self.match_spans_exact(true_spans, pred_spans)
            overall_metrics = self.calculate_metrics(total_tp, total_fp, total_fn)
            
            return {
                'original_text': example.unmasked_text,
                'true_masked': example.masked_text,
                'pred_masked': prediction.masked_text,
                'entities': prediction.entities,
                'true_spans': [(s.entity_type, s.start, s.end, s.text) for s in true_spans],
                'pred_spans': [(s.entity_type, s.start, s.end, s.text) for s in pred_spans],
                'per_class_metrics': per_class_metrics,
                'overall_metrics': overall_metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            return {
                'original_text': example.unmasked_text,
                'error': str(e),
                'success': False
            }
    
    def evaluate_dataset(self, 
                        examples: List[PIIExample],
                        predictions: List[PIIPrediction],
                        experiment_name: str = "custom_experiment",
                        model_name: str = "model",
                        config: Dict = None) -> CustomEvaluationResult:
        """
        Evaluate predictions on a full dataset using exact span matching.
        
        Args:
            examples: List of ground truth PIIExample objects
            predictions: List of PIIPrediction objects with pre-computed spans
            experiment_name: Name of the experiment
            model_name: Name of the model being evaluated
            config: Configuration dictionary
            
        Returns:
            CustomEvaluationResult object with comprehensive metrics
        """
        if len(examples) != len(predictions):
            raise ValueError(f"Mismatch: {len(examples)} examples vs {len(predictions)} predictions")
        
        logger.info(f"Evaluating {len(examples)} examples for {experiment_name}")
        
        # Aggregate metrics
        per_class_aggregated = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        total_tp = 0
        total_fp = 0
        total_fn = 0
        successful_predictions = 0
        detailed_results = []
        
        # Evaluate each example
        for example, prediction in zip(examples, predictions):
            result = self.evaluate_single_example(example, prediction)
            detailed_results.append(result)
            
            if result['success']:
                successful_predictions += 1
                
                # Aggregate per-class metrics
                for entity_type, metrics in result['per_class_metrics'].items():
                    per_class_aggregated[entity_type]['tp'] += metrics['true_positives']
                    per_class_aggregated[entity_type]['fp'] += metrics['false_positives']
                    per_class_aggregated[entity_type]['fn'] += metrics['false_negatives']
                
                # Aggregate overall metrics
                overall = result['overall_metrics']
                total_tp += overall['true_positives']
                total_fp += overall['false_positives']
                total_fn += overall['false_negatives']
        
        # Calculate final per-class metrics
        per_class_metrics = {}
        for entity_type, counts in per_class_aggregated.items():
            per_class_metrics[entity_type] = self.calculate_metrics(
                counts['tp'], counts['fp'], counts['fn']
            )
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        return CustomEvaluationResult(
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
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
    
    def print_evaluation_report(self, result: CustomEvaluationResult):
        """Print a comprehensive evaluation report."""
        
        print(f"\n{'='*80}")
        print(f"📊 CUSTOM PII EVALUATION REPORT: {result.experiment_name}")
        print(f"🤖 Model: {result.model_name}")
        print(f"{'='*80}")
        
        # Global metrics
        print(f"\n🎯 OVERALL METRICS:")
        print(f"   Precision:     {result.precision:.3f}")
        print(f"   Recall:        {result.recall:.3f}")
        print(f"   F1-Score:      {result.f1_score:.3f}")
        
        # Summary stats
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total Examples:        {result.total_examples}")
        print(f"   Successful Predictions: {result.successful_predictions}")
        print(f"   Success Rate:          {result.successful_predictions/result.total_examples:.3f}")
        print(f"   Total True Entities:   {result.total_true_entities}")
        print(f"   Total Pred Entities:   {result.total_pred_entities}")
        print(f"   Correct Entities:      {result.total_correct_entities}")
        
        # Per-class metrics
        if result.per_class_metrics:
            print(f"\n📋 PER-CLASS METRICS:")
            print(f"{'Entity Type':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 70)
            
            for entity_type, metrics in sorted(result.per_class_metrics.items()):
                print(f"{entity_type:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['f1_score']:<10.3f} {metrics['support']:<10}")
        
        # Configuration
        if result.config:
            print(f"\n⚙️  CONFIGURATION:")
            for key, value in result.config.items():
                print(f"   {key}: {value}")
        
        print(f"\n{'='*80}") 