"""
Text processing utilities for PII masking.

This module contains shared functionality for processing PII predictions
and reconstructing masked text. It's designed to be used by both the
base model classes and the evaluator.
"""

import re
import json
import logging
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EntitySpan:
    """Represents an entity span with position and type."""
    entity_type: str
    start: int
    end: int
    text: str

@dataclass
class PIIPrediction:
    """
    Complete PII prediction with entities, spans, and masked text.
    
    This is the enriched version that contains all computed information:
    - entities: Raw entity dictionary from JSON
    - spans: Computed EntitySpan objects with positions
    - masked_text: Pre-computed masked text
    - original_text: Original text used for computation
    """
    entities: Dict[str, List[str]]
    spans: List[EntitySpan]
    masked_text: str
    original_text: str
    
    @classmethod
    def from_json_and_text(cls, json_str: str, original_text: str) -> 'PIIPrediction':
        """
        Create complete PIIPrediction from JSON string and original text.
        
        This is the main factory method that parses JSON, computes spans,
        and reconstructs masked text in one go.
        
        Args:
            json_str: JSON string from model prediction
            original_text: Original unmasked text
            
        Returns:
            Complete PIIPrediction with all fields populated
        """
        entities = parse_json_prediction(json_str)
        spans = json_to_spans(original_text, entities)
        masked_text = reconstruct_masked_text(original_text, entities)
        
        return cls(
            entities=entities,
            spans=spans,
            masked_text=masked_text,
            original_text=original_text
        )
    
    @classmethod
    def from_json_string(cls, json_str: str) -> 'PIIPrediction':
        """
        Create PIIPrediction from JSON string only (legacy method).
        
        Note: This creates an incomplete object without spans or masked_text.
        Use from_json_and_text() for complete objects.
        """
        entities = parse_json_prediction(json_str)
        return cls(
            entities=entities,
            spans=[],
            masked_text="",
            original_text=""
        )
    
    def to_json_string(self) -> str:
        """Convert to JSON string format."""
        return json.dumps({"PII": self.entities}, ensure_ascii=False)
    
    def is_empty(self) -> bool:
        """Check if prediction contains no entities."""
        return not self.entities or all(not v for v in self.entities.values())
    
    def get_spans_by_type(self, entity_type: str) -> List[EntitySpan]:
        """Get all spans of a specific entity type."""
        return [span for span in self.spans if span.entity_type == entity_type]
    
    def get_all_entity_types(self) -> List[str]:
        """Get all unique entity types in this prediction."""
        return list(self.entities.keys())

def parse_json_prediction(prediction: str) -> Dict[str, List[str]]:
    """
    Parse JSON prediction from model output.
    
    Since we use Mistral's JSON mode, the output should be valid JSON.
    
    Args:
        prediction: JSON string from model (guaranteed valid JSON)
        
    Returns:
        Dictionary with entity types as keys and lists of substrings as values
    """
    try:
        parsed = json.loads(prediction.strip())
        
        if 'PII' in parsed:
            return parsed['PII']
        else:
            return parsed
            
    except Exception as e:
        logger.error(f"Error parsing JSON prediction: {e}")
        logger.error(f"Prediction was: {prediction}")
        return {}

def json_to_spans(text: str, pii_dict: Dict[str, List[str]]) -> List[EntitySpan]:
    """
    Convert JSON PII dictionary to entity spans using regex matching.
    
    Args:
        text: Original text to search in
        pii_dict: Dictionary with entity types and their substrings
        
    Returns:
        List of EntitySpan objects with positions in the text
    """
    spans = []
    
    for entity_type, substrings in pii_dict.items():
        for substring in substrings:
            if not substring or not isinstance(substring, str):
                continue
            
            escaped_substring = re.escape(substring)
            
            for match in re.finditer(escaped_substring, text):
                span = EntitySpan(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    text=substring
                )
                spans.append(span)
    
    spans.sort(key=lambda x: x.start)
    
    return spans

def reconstruct_masked_text(text: str, pii_dict: Dict[str, List[str]]) -> str:
    """
    Reconstruct masked text by replacing PII substrings with placeholders.
    
    Args:
        text: Original text
        pii_dict: Dictionary with entity types and their substrings
        
    Returns:
        Text with PII replaced by [ENTITY_TYPE_X] placeholders
    """
    masked_text = text
    
    all_spans = json_to_spans(text, pii_dict)
    all_spans.sort(key=lambda x: x.start)
    
    entity_counters = defaultdict(int)
    span_numbers = {}
    
    for i, span in enumerate(all_spans):
        entity_counters[span.entity_type] += 1
        span_key = (span.start, span.end, span.entity_type)
        span_numbers[span_key] = entity_counters[span.entity_type]
    
    all_spans.sort(key=lambda x: x.start, reverse=True)
    
    for span in all_spans:
        span_key = (span.start, span.end, span.entity_type)
        number = span_numbers[span_key]
        placeholder = f"[{span.entity_type}_{number}]"
        masked_text = masked_text[:span.start] + placeholder + masked_text[span.end:]
    
    return masked_text


def reconstruct_masked_text_from_prediction(text: str, prediction: Union[str, Any]) -> str:
    """
    Reconstruct masked text from a prediction (JSON string or PIIPrediction object).
    
    Args:
        text: Original unmasked text
        prediction: Either a JSON string or PIIPrediction object
        
    Returns:
        Text with PII replaced by [ENTITY_TYPE_X] placeholders
    """
    if isinstance(prediction, str):
        pii_dict = parse_json_prediction(prediction)
    else:
        pii_dict = prediction.entities
        
    return reconstruct_masked_text(text, pii_dict) 