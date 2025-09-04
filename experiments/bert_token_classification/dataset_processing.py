#!/usr/bin/env python3
"""
Dataset processing for BERT token classification.

This script processes the PII dataset to create properly aligned token-level labels
for BERT fine-tuning. It handles the alignment between tokenized text and span labels,
ensuring each token gets the correct PII label based on its intersection with entity spans.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pii_masking import PIIDataLoader, PIIExample
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenClassificationExample:
    """
    A training example for token classification.
    """
    tokens: List[str]
    labels: List[str]
    original_text: str
    token_spans: List[Tuple[int, int]]

class PIITokenClassificationProcessor:
    """
    Processes PII examples into token classification format for BERT training.
    """
    
    def __init__(self, tokenizer_name: str = "distilbert-base-uncased"):
        """
        Initialize processor with tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_set = set()
        
    def align_tokens_with_spans(self, text: str, span_labels: List[List]) -> TokenClassificationExample:
        """
        Align tokenized text with PII span labels.
        
        This is the core function that:
        1. Tokenizes the text with BERT tokenizer
        2. Gets character-level spans for each token
        3. Determines PII label for each token based on intersection with entity spans
        
        Args:
            text: Original text
            span_labels: List of [start, end, label] spans
            
        Returns:
            TokenClassificationExample with aligned tokens and labels
        """
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        token_spans = encoding['offset_mapping']
        
        span_lookup = self._create_span_lookup(span_labels)
        
        labels = []
        for i, (token, (start, end)) in enumerate(zip(tokens, token_spans)):
            if token in ['[CLS]', '[SEP]'] or start == end == 0:
                labels.append('O')
            else:
                label = self._get_token_label(start, end, span_lookup)
                labels.append(label)
                self.label_set.add(label)
        
        return TokenClassificationExample(
            tokens=tokens,
            labels=labels,
            original_text=text,
            token_spans=token_spans
        )
    
    def _create_span_lookup(self, span_labels: List[List]) -> Dict[Tuple[int, int], str]:
        """
        Create a lookup dictionary for span labels.
        
        Args:
            span_labels: List of [start, end, label] spans
            
        Returns:
            Dictionary mapping (start, end) to label
        """
        span_lookup = {}
        for span in span_labels:
            if len(span) >= 3:
                start, end, label = span[0], span[1], span[2]
                clean_label = self._clean_label(label)
                if clean_label != 'O':
                    span_lookup[(start, end)] = clean_label
        return span_lookup
    
    def _clean_label(self, label: str) -> str:
        """
        Clean PII label by removing BIO prefixes and suffixes.
        
        Args:
            label: Original label (e.g., 'B-FIRSTNAME_1', 'I-EMAIL')
            
        Returns:
            Clean label (e.g., 'FIRSTNAME', 'EMAIL')
        """
        label = label.replace('B-', '').replace('I-', '')
        
        if '_' in label:
            label = label.split('_')[0]
        
        return label
    
    def _get_token_label(self, token_start: int, token_end: int, 
                        span_lookup: Dict[Tuple[int, int], str]) -> str:
        """
        Get PII label for a token based on its intersection with entity spans.
        
        A token gets a PII label if it has ANY intersection with a PII span.
        If it intersects with multiple spans, we take the first one found.
        
        Args:
            token_start: Token start position
            token_end: Token end position  
            span_lookup: Dictionary of span positions to labels
            
        Returns:
            PII label or 'O' if no intersection
        """
        for (span_start, span_end), label in span_lookup.items():
            if self._spans_intersect(token_start, token_end, span_start, span_end):
                return label
        
        return 'O'
    
    def _spans_intersect(self, t_start: int, t_end: int, s_start: int, s_end: int) -> bool:
        """
        Check if two spans intersect.
        
        Args:
            t_start, t_end: Token span
            s_start, s_end: Entity span
            
        Returns:
            True if spans intersect
        """
        return not (t_end <= s_start or s_end <= t_start)
    
    def process_examples(self, examples: List[PIIExample]) -> List[TokenClassificationExample]:
        """
        Process a list of PII examples into token classification format.
        
        Args:
            examples: List of PIIExample objects
            
        Returns:
            List of TokenClassificationExample objects
        """
        processed_examples = []
        
        for i, example in enumerate(examples):
            try:
                processed = self.align_tokens_with_spans(
                    example.unmasked_text, 
                    example.span_labels
                )
                processed_examples.append(processed)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(examples)} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_examples)} examples")
        logger.info(f"Found {len(self.label_set)} unique labels: {sorted(self.label_set)}")
        
        return processed_examples
    
    def create_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create label to ID mappings for model training.
        
        Returns:
            Tuple of (label2id, id2label) dictionaries
        """
        sorted_labels = sorted(self.label_set)
        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        return label2id, id2label
    
    def save_processed_dataset(self, examples: List[TokenClassificationExample], 
                              output_path: str):
        """
        Save processed examples to JSON file.
        
        Args:
            examples: Processed examples
            output_path: Output file path
        """
        data = []
        for example in examples:
            data.append({
                'tokens': example.tokens,
                'labels': example.labels,
                'original_text': example.original_text,
                'token_spans': example.token_spans
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(examples)} processed examples to {output_path}")

def main():
    """Main processing function."""
    dataset_path = "../../data/english_pii_43k.jsonl"
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    tokenizer_name = "distilbert-base-uncased"
    max_samples = None
    
    logger.info(f"Starting dataset processing with tokenizer: {tokenizer_name}")
    
    data_loader = PIIDataLoader(data_dir=Path(dataset_path).parent)
    examples = data_loader.load_dataset(
        language='english',
        max_samples=max_samples,
        shuffle=False
    )
    
    if not examples:
        raise ValueError("No examples loaded from dataset")
    
    logger.info(f"Loaded {len(examples)} examples from dataset")
    
    processor = PIITokenClassificationProcessor(tokenizer_name=tokenizer_name)
    processed_examples = processor.process_examples(examples)
    
    label2id, id2label = processor.create_label_mappings()
    
    processor.save_processed_dataset(
        processed_examples, 
        output_dir / "processed_examples.json"
    )
    
    with open(output_dir / "label_mappings.json", 'w') as f:
        json.dump({
            'label2id': label2id,
            'id2label': id2label,
            'num_labels': len(label2id)
        }, f, indent=2)
    
    logger.info("Processing completed successfully!")
    logger.info(f"Total processed examples: {len(processed_examples)}")
    logger.info(f"Number of unique labels: {len(label2id)}")
    logger.info(f"Labels: {sorted(label2id.keys())}")
    
    if processed_examples:
        example = processed_examples[0]
        logger.info(f"\nExample processed output:")
        logger.info(f"Original text: {example.original_text[:100]}...")
        logger.info(f"Tokens: {example.tokens[:10]}...")
        logger.info(f"Labels: {example.labels[:10]}...")

if __name__ == "__main__":
    main() 