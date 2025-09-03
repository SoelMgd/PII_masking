#!/usr/bin/env python3
"""
Dataset processing for Mistral fine-tuning.

This script:
1. Loads English and French PII datasets
2. Converts them to Mistral fine-tuning format (messages with user/assistant)
3. Creates training and validation splits
4. Saves the processed datasets for fine-tuning
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pii_masking import PIIDataLoader, PIIExample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PIIDatasetProcessor:
    """Processor for converting PII datasets to Mistral fine-tuning format."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the dataset processor.
        
        Args:
            data_dir: Directory containing the PII datasets
        """
        self.data_dir = Path(data_dir)
        self.data_loader = PIIDataLoader(data_dir=self.data_dir)
    
    def example_to_mistral_format(self, example: PIIExample) -> Dict[str, Any]:
        """
        Convert a PIIExample to Mistral fine-tuning format.
        
        Args:
            example: PIIExample object
            
        Returns:
            Dictionary in Mistral fine-tuning format
        """
        # Create the user prompt
        user_prompt = f"""Please extract all Personal Identifiable Information (PII) from the text.
Text to analyze:
{example.unmasked_text}"""

        # Create the assistant response (ground truth JSON)
        pii_dict = self._spans_to_json(example.unmasked_text, example.span_labels)
        assistant_response = json.dumps(pii_dict, ensure_ascii=False)
        
        # Return in Mistral format
        return {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant", 
                    "content": assistant_response
                }
            ]
        }
    
    def _spans_to_json(self, text: str, span_labels: List[List]) -> Dict[str, Dict[str, List[str]]]:
        """
        Convert span labels to JSON format.
        
        Args:
            text: Original text
            span_labels: List of [start, end, label] spans
            
        Returns:
            Dictionary in {"PII": {"ENTITY_TYPE": ["substring1", ...]}} format
        """
        pii_dict = {}
        
        for span in span_labels:
            if len(span) >= 3:
                start, end, label = span[0], span[1], span[2]
                
                # Remove BIO prefixes and suffixes
                entity_type = label.replace('B-', '').replace('I-', '')
                if '_' in entity_type:
                    entity_type = entity_type.split('_')[0]
                
                # Skip "O" labels (non-PII text)
                if entity_type == 'O':
                    continue
                
                # Extract substring
                substring = text[start:end]
                
                if entity_type not in pii_dict:
                    pii_dict[entity_type] = []
                
                # Avoid duplicates
                if substring not in pii_dict[entity_type]:
                    pii_dict[entity_type].append(substring)
        
        return {"PII": pii_dict}
    
    def load_and_process_dataset(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Load and process the English PII dataset.
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            List of examples in Mistral fine-tuning format
        """
        logger.info("Loading English PII dataset...")
        
        # Load examples
        examples = self.data_loader.load_dataset(
            language='english',
            max_samples=max_samples,
            shuffle=True,
            seed=42
        )
        
        if not examples:
            logger.warning("No examples loaded from English dataset")
            return []
        
        logger.info(f"Loaded {len(examples)} examples from English dataset")
        
        # Convert to Mistral format
        processed_examples = []
        for example in examples:
            try:
                mistral_example = self.example_to_mistral_format(example)
                processed_examples.append(mistral_example)
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_examples)} examples")
        return processed_examples
    
    def create_train_val_split(self, examples: List[Dict[str, Any]], val_ratio: float = 0.1) -> tuple:
        """
        Split examples into training and validation sets.
        
        Args:
            examples: List of examples in Mistral format
            val_ratio: Ratio of validation data (default 0.1 = 10%)
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        # Shuffle examples
        random.shuffle(examples)
        
        # Calculate split point
        val_size = int(len(examples) * val_ratio)
        train_size = len(examples) - val_size
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:]
        
        logger.info(f"Split: {len(train_examples)} training, {len(val_examples)} validation")
        return train_examples, val_examples
    
    def save_jsonl(self, examples: List[Dict[str, Any]], output_path: Path):
        """
        Save examples to JSONL format.
        
        Args:
            examples: List of examples to save
            output_path: Path to save the JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def get_dataset_stats(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the processed dataset.
        
        Args:
            examples: List of processed examples
            
        Returns:
            Dictionary with dataset statistics
        """
        total_examples = len(examples)
        entity_counts = defaultdict(int)
        total_entities = 0
        
        for example in examples:
            # Parse assistant response to count entities
            try:
                assistant_content = example['messages'][1]['content']
                pii_data = json.loads(assistant_content)
                
                if 'PII' in pii_data:
                    for entity_type, entities in pii_data['PII'].items():
                        entity_counts[entity_type] += len(entities)
                        total_entities += len(entities)
            except Exception as e:
                logger.warning(f"Error parsing example for stats: {e}")
                continue
        
        return {
            'total_examples': total_examples,
            'total_entities': total_entities,
            'avg_entities_per_example': total_entities / total_examples if total_examples > 0 else 0,
            'entity_type_counts': dict(entity_counts),
            'unique_entity_types': len(entity_counts)
        }

def main():
    """Main function for dataset processing."""
    parser = argparse.ArgumentParser(description="Process PII datasets for Mistral fine-tuning")
    
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Directory containing PII datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="../data/fine_tuning",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PIIDatasetProcessor(data_dir=Path(args.data_dir))
    output_dir = Path(args.output_dir)
    
    # Process English dataset
    all_examples = processor.load_and_process_dataset(max_samples=args.max_samples)
    
    if not all_examples:
        logger.error("No examples processed. Check your data directory and files.")
        return
    
    # Shuffle dataset
    random.shuffle(all_examples)
    logger.info(f"Processed dataset: {len(all_examples)} examples")
    
    # Get dataset stats
    dataset_stats = processor.get_dataset_stats(all_examples)
    logger.info(f"Dataset stats: {dataset_stats}")
    
    # Create train/val split
    train_examples, val_examples = processor.create_train_val_split(
        all_examples, 
        val_ratio=args.val_ratio
    )
    
    # Save datasets
    processor.save_jsonl(train_examples, output_dir / "train.jsonl")
    processor.save_jsonl(val_examples, output_dir / "validation.jsonl")
    
    # Save statistics
    stats = {
        'dataset_stats': dataset_stats,
        'train_size': len(train_examples),
        'val_size': len(val_examples),
        'val_ratio': args.val_ratio
    }
    
    with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset processing completed!")
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    logger.info(f"Files saved to: {output_dir}")
    
    print(f"\nDataset processing completed successfully!")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Files saved to: {output_dir}")
    print(f"Average entities per example: {dataset_stats['avg_entities_per_example']:.2f}")
    print(f"Unique entity types: {dataset_stats['unique_entity_types']}")

if __name__ == "__main__":
    main() 