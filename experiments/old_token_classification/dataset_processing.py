#!/usr/bin/env python3
"""
Dataset processing for token classification with Mistral.

This script:
1. Loads PII datasets and converts them to token classification format
2. Tokenizes text using Mistral tokenizer
3. Aligns PII labels with tokens
4. Creates PyTorch-compatible datasets for training
5. Handles both English and French datasets
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
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pii_masking import PIIDataLoader, PIIExample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenClassificationExample:
    """Example for token classification."""
    text: str
    tokens: List[str]
    token_ids: List[int]  # Token IDs from Tekken v3
    labels: List[str]
    token_positions: List[Tuple[int, int]]  # (start, end) positions in original text

class TokenClassificationProcessor:
    """Processor for converting PII datasets to token classification format."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing the PII datasets
        """
        self.data_dir = Path(data_dir)
        self.data_loader = PIIDataLoader(data_dir=self.data_dir)
        self.tokenizer = None
        self._init_tokenizer()
        
        # Label mapping
        self.label_to_id = {"O": 0}  # "O" for non-PII tokens
        self.id_to_label = {0: "O"}
        self.next_label_id = 1
    
    def _init_tokenizer(self):
        """Initialize Mistral Tekken v3 tokenizer."""
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            # Use Tekken v3 tokenizer for Ministral-8B-Instruct-2410
            self.tokenizer = MistralTokenizer.v3(is_tekken=True)
            logger.info("Mistral Tekken v3 tokenizer initialized successfully")
        except ImportError:
            logger.error("Failed to import mistral_common. Install with: pip install mistral-common tiktoken")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Mistral Tekken tokenizer: {e}")
            raise
    
    def _get_label_id(self, label: str) -> int:
        """Get or create label ID."""
        if label not in self.label_to_id:
            self.label_to_id[label] = self.next_label_id
            self.id_to_label[self.next_label_id] = label
            self.next_label_id += 1
        return self.label_to_id[label]
    
    def _tokenize_with_positions(self, text: str) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
        """
        Tokenize text with Tekken v3 and return tokens, token IDs, and their exact positions.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tuple of (tokens, token_ids, positions) where positions are (start, end) tuples
        """
        try:
            # Use Mistral Tekken tokenizer for raw text tokenization
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            
            # Create a minimal chat completion request with just the text
            request = ChatCompletionRequest(
                messages=[UserMessage(content=text)],
                model="ministral"  # Model name for Tekken
            )
            
            # Encode the request
            encoded_result = self.tokenizer.encode_chat_completion(request)
            token_ids = encoded_result.tokens
            
            # Decode each token individually
            raw_tokens = []
            for token_id in token_ids:
                token_text = self.tokenizer.decode([token_id])
                raw_tokens.append(token_text)
            
            # Remove the first 2 empty tokens that Tekken always adds
            if len(raw_tokens) >= 2 and raw_tokens[0] == '' and raw_tokens[1] == '':
                tokens = raw_tokens[2:]
                cleaned_token_ids = token_ids[2:]  # Also remove from token IDs
                logger.debug(f"Removed 2 initial empty tokens, kept {len(tokens)} tokens")
            else:
                tokens = raw_tokens
                cleaned_token_ids = token_ids
                logger.warning(f"Expected 2 initial empty tokens, but got: {raw_tokens[:3]}")
            
            # Reconstruct the text from tokens to verify alignment
            reconstructed = ''.join(tokens)
            if reconstructed != text:
                logger.warning(f"Text reconstruction mismatch!")
                logger.warning(f"Original:      '{text}'")
                logger.warning(f"Reconstructed: '{reconstructed}'")
            
            # Calculate exact positions by concatenating tokens
            positions = []
            current_pos = 0
            
            for token in tokens:
                start = current_pos
                end = current_pos + len(token)
                positions.append((start, end))
                current_pos = end
            
            logger.debug(f"Tokenized '{text[:50]}...' into {len(tokens)} tokens")
            return tokens, cleaned_token_ids, positions
            
        except Exception as e:
            logger.warning(f"Error with Mistral Tekken tokenizer: {e}, falling back to simple whitespace tokenization")
            # Fallback to simple tokenization
            words = text.split()
            tokens = []
            token_ids = []
            positions = []
            current_pos = 0
            
            for word in words:
                # Find the word in the text starting from current_pos
                start = text.find(word, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(word)
                
                tokens.append(word)
                token_ids.append(1)  # Dummy token ID for fallback
                positions.append((start, end))
                current_pos = end
            
            return tokens, token_ids, positions
    
    def _extract_entity_spans(self, span_labels: List[List]) -> List[Tuple[int, int, str]]:
        """
        Extract entity spans from dataset span_labels.
        
        Args:
            span_labels: Raw span labels from dataset
            
        Returns:
            List of (start, end, entity_type) tuples
        """
        entity_spans = []
        
        for span in span_labels:
            if len(span) >= 3:
                start, end, label = span[0], span[1], span[2]
                
                # Clean up label (remove BIO prefixes and suffixes)
                clean_label = label.replace('B-', '').replace('I-', '')
                if '_' in clean_label:
                    clean_label = clean_label.split('_')[0]
                
                # Skip "O" labels
                if clean_label != 'O':
                    entity_spans.append((start, end, clean_label))
        
        return entity_spans
    
    def _align_spans_with_tokens(self, entity_spans: List[Tuple[int, int, str]], 
                                token_positions: List[Tuple[int, int]]) -> List[Tuple[int, int, str]]:
        """
        Align entity spans with token boundaries.
        
        Args:
            entity_spans: List of (start, end, entity_type) from dataset
            token_positions: Token positions in original text
            
        Returns:
            List of aligned (start, end, entity_type) at token boundaries
        """
        aligned_spans = []
        
        for entity_start, entity_end, entity_type in entity_spans:
            # Find tokens that overlap with this entity
            overlapping_tokens = []
            
            for i, (token_start, token_end) in enumerate(token_positions):
                # Check if token overlaps with entity span
                if (token_start < entity_end and token_end > entity_start):
                    overlapping_tokens.append(i)
            
            if overlapping_tokens:
                # Extend span to cover all overlapping tokens
                first_token_idx = overlapping_tokens[0]
                last_token_idx = overlapping_tokens[-1]
                
                aligned_start = token_positions[first_token_idx][0]
                aligned_end = token_positions[last_token_idx][1]
                
                aligned_spans.append((aligned_start, aligned_end, entity_type))
                
                logger.debug(f"Aligned entity {entity_type}: [{entity_start}:{entity_end}] -> [{aligned_start}:{aligned_end}] (tokens {first_token_idx}-{last_token_idx})")
        
        return aligned_spans
    
    def _assign_labels_to_tokens(self, aligned_spans: List[Tuple[int, int, str]], 
                                token_positions: List[Tuple[int, int]]) -> List[str]:
        """
        Assign labels to tokens based on aligned spans.
        
        Args:
            aligned_spans: List of (start, end, entity_type) aligned to token boundaries
            token_positions: Token positions in original text
            
        Returns:
            List of labels for each token
        """
        # Initialize all tokens as "O" (non-PII)
        token_labels = ["O"] * len(token_positions)
        
        # Assign labels based on aligned spans
        for span_start, span_end, entity_type in aligned_spans:
            for i, (token_start, token_end) in enumerate(token_positions):
                # Check if token is completely within the aligned span
                if token_start >= span_start and token_end <= span_end:
                    token_labels[i] = entity_type
        
        return token_labels
    
    def process_example(self, example: PIIExample) -> TokenClassificationExample:
        """
        Process a single PIIExample into token classification format.
        
        Args:
            example: PIIExample from dataset
            
        Returns:
            TokenClassificationExample
        """
        # Tokenize the text with Tekken v3
        tokens, token_ids, token_positions = self._tokenize_with_positions(example.unmasked_text)
        
        # Extract entity spans from dataset
        entity_spans = self._extract_entity_spans(example.span_labels)
        
        # Align entity spans with token boundaries
        aligned_spans = self._align_spans_with_tokens(entity_spans, token_positions)
        
        # Assign labels to tokens based on aligned spans
        labels = self._assign_labels_to_tokens(aligned_spans, token_positions)
        
        logger.debug(f"Processed example: {len(tokens)} tokens, {len(entity_spans)} entities -> {len(aligned_spans)} aligned spans")
        
        return TokenClassificationExample(
            text=example.unmasked_text,
            tokens=tokens,
            token_ids=token_ids,
            labels=labels,
            token_positions=token_positions
        )
    
    def process_dataset(self, language: str, max_samples: int = None) -> List[TokenClassificationExample]:
        """
        Process a complete dataset.
        
        Args:
            language: Language of the dataset ('english' or 'french')
            max_samples: Maximum number of samples to process
            
        Returns:
            List of TokenClassificationExample objects
        """
        logger.info(f"Processing {language} dataset...")
        
        # Load examples
        examples = self.data_loader.load_dataset(
            language=language,
            max_samples=max_samples,
            shuffle=True,
            seed=42
        )
        
        if not examples:
            logger.warning(f"No examples loaded for {language}")
            return []
        
        logger.info(f"Loaded {len(examples)} examples for {language}")
        
        # Process each example
        processed_examples = []
        for i, example in enumerate(examples):
            try:
                processed_example = self.process_example(example)
                processed_examples.append(processed_example)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(examples)} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_examples)} examples for {language}")
        return processed_examples
    
    def create_pytorch_dataset(self, examples: List[TokenClassificationExample]) -> Dict[str, Any]:
        """
        Create PyTorch-compatible dataset.
        
        Args:
            examples: List of processed examples
            
        Returns:
            Dictionary with dataset components
        """
        texts = []
        all_tokens = []
        all_token_ids = []
        all_labels = []
        all_label_ids = []
        
        for example in examples:
            texts.append(example.text)
            all_tokens.append(example.tokens)
            all_token_ids.append(example.token_ids)
            all_labels.append(example.labels)
            
            # Convert labels to IDs
            label_ids = [self._get_label_id(label) for label in example.labels]
            all_label_ids.append(label_ids)
        
        return {
            'texts': texts,
            'tokens': all_tokens,
            'token_ids': all_token_ids,  # ✅ NOUVEAU : Token IDs pré-calculés
            'labels': all_labels,
            'label_ids': all_label_ids,
            'label_to_id': self.label_to_id.copy(),
            'id_to_label': self.id_to_label.copy(),
            'num_labels': len(self.label_to_id)
        }
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: Path):
        """Save dataset to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Saved dataset to {output_path}")
    
    def get_dataset_stats(self, examples: List[TokenClassificationExample]) -> Dict[str, Any]:
        """Get statistics about the processed dataset."""
        total_examples = len(examples)
        total_tokens = sum(len(ex.tokens) for ex in examples)
        
        # Count labels
        label_counts = defaultdict(int)
        for example in examples:
            for label in example.labels:
                label_counts[label] += 1
        
        # Calculate average sequence length
        avg_seq_length = total_tokens / total_examples if total_examples > 0 else 0
        
        return {
            'total_examples': total_examples,
            'total_tokens': total_tokens,
            'avg_sequence_length': avg_seq_length,
            'label_counts': dict(label_counts),
            'unique_labels': len(label_counts),
            'non_o_ratio': (total_tokens - label_counts.get('O', 0)) / total_tokens if total_tokens > 0 else 0
        }

def main():
    """Main function for dataset processing."""
    parser = argparse.ArgumentParser(description="Process PII datasets for token classification")
    
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Directory containing PII datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--max-english",
        type=int,
        default=None,
        help="Maximum number of English examples to process"
    )
    parser.add_argument(
        "--max-french", 
        type=int,
        default=None,
        help="Maximum number of French examples to process"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Process only English dataset"
    )
    parser.add_argument(
        "--french-only", 
        action="store_true",
        help="Process only French dataset"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = TokenClassificationProcessor(data_dir=Path(args.data_dir))
    output_dir = Path(args.output_dir)
    
    # Process datasets
    all_examples = []
    
    if not args.french_only:
        # Process English dataset
        english_examples = processor.process_dataset(
            language='english',
            max_samples=args.max_english
        )
        all_examples.extend(english_examples)
        
        # Get English stats
        english_stats = processor.get_dataset_stats(english_examples)
        logger.info(f"English dataset stats: {english_stats}")
    
    if not args.english_only:
        # Process French dataset
        french_examples = processor.process_dataset(
            language='french',
            max_samples=args.max_french
        )
        all_examples.extend(french_examples)
        
        # Get French stats
        french_stats = processor.get_dataset_stats(french_examples)
        logger.info(f"French dataset stats: {french_stats}")
    
    if not all_examples:
        logger.error("No examples processed. Check your data directory and files.")
        return
    
    # Shuffle combined dataset
    import random
    random.shuffle(all_examples)
    logger.info(f"Combined dataset: {len(all_examples)} examples")
    
    # Get combined stats
    combined_stats = processor.get_dataset_stats(all_examples)
    logger.info(f"Combined dataset stats: {combined_stats}")
    
    # Create train/val split
    val_size = int(len(all_examples) * args.val_ratio)
    train_examples = all_examples[:-val_size] if val_size > 0 else all_examples
    val_examples = all_examples[-val_size:] if val_size > 0 else []
    
    logger.info(f"Split: {len(train_examples)} training, {len(val_examples)} validation")
    
    # Create PyTorch datasets
    train_dataset = processor.create_pytorch_dataset(train_examples)
    val_dataset = processor.create_pytorch_dataset(val_examples) if val_examples else None
    
    # Save datasets
    processor.save_dataset(train_dataset, output_dir / "train_dataset.pkl")
    if val_dataset:
        processor.save_dataset(val_dataset, output_dir / "val_dataset.pkl")
    
    # Save label mappings separately for easy access
    label_info = {
        'label_to_id': processor.label_to_id,
        'id_to_label': processor.id_to_label,
        'num_labels': len(processor.label_to_id)
    }
    
    with open(output_dir / "label_mappings.json", 'w') as f:
        json.dump(label_info, f, indent=2)
    
    # Save statistics
    stats = {
        'combined_stats': combined_stats,
        'train_size': len(train_examples),
        'val_size': len(val_examples),
        'val_ratio': args.val_ratio,
        'label_info': label_info
    }
    
    if not args.french_only:
        stats['english_stats'] = english_stats
    if not args.english_only:
        stats['french_stats'] = french_stats
    
    with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset processing completed!")
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    logger.info(f"Files saved to: {output_dir}")
    
    print(f"\n🎉 Token classification dataset processing completed!")
    print(f"📊 Training examples: {len(train_examples)}")
    print(f"📊 Validation examples: {len(val_examples)}")
    print(f"📁 Files saved to: {output_dir}")
    print(f"🏷️  Number of labels: {len(processor.label_to_id)}")
    print(f"📈 Average sequence length: {combined_stats['avg_sequence_length']:.1f}")
    print(f"📊 Non-O token ratio: {combined_stats['non_o_ratio']:.3f}")

if __name__ == "__main__":
    main() 