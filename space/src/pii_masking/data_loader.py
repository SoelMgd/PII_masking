"""
Data loading utilities for PII masking datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PIIExample:
    """
    Represents a single PII example from the dataset.
    
    Attributes:
        masked_text: Text with PII replaced by placeholders
        unmasked_text: Original text containing PII
        privacy_mask: Mapping from placeholders to actual PII values
        span_labels: List of [start, end, entity_type] for each span
        bio_labels: BIO tags for each token
        tokenised_text: List of tokens
    """
    masked_text: str
    unmasked_text: str
    privacy_mask: Dict[str, str]
    span_labels: List[List]
    bio_labels: List[str]
    tokenised_text: List[str]
    
    def get_entity_types(self) -> List[str]:
        """Extract unique entity types from privacy_mask."""
        import re
        entity_types = []
        for placeholder in self.privacy_mask.keys():
            match = re.match(r'\[([A-Z_]+)_\d+\]', placeholder)
            if match:
                entity_types.append(match.group(1))
        return list(set(entity_types))
    
    def get_pii_spans(self) -> List[Dict]:
        """
        Extract PII spans with their positions and types.
        
        Returns:
            List of dicts with keys: start, end, entity_type, text
        """
        spans = []
        for span in self.span_labels:
            if len(span) >= 3 and span[2] != 'O':
                # Remove B- or I- prefix if present
                entity_type = span[2].replace('B-', '').replace('I-', '')
                spans.append({
                    'start': span[0],
                    'end': span[1], 
                    'entity_type': entity_type,
                    'text': self.unmasked_text[span[0]:span[1]] if span[0] < len(self.unmasked_text) else ""
                })
        return spans

class PIIDataLoader:
    """
    Handles loading and preprocessing of PII datasets.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.supported_languages = ['english', 'french']
        
    def load_dataset(self, 
                    language: str = 'english',
                    max_samples: Optional[int] = None,
                    shuffle: bool = False,
                    seed: int = 42) -> List[PIIExample]:
        """
        Load dataset for a specific language.
        
        Args:
            language: Language of the dataset ('english', 'french', etc.)
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            
        Returns:
            List of PIIExample objects
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language '{language}' not supported. "
                           f"Available: {self.supported_languages}")
        
        # Find dataset file
        dataset_file = self._find_dataset_file(language)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        logger.info(f"Loading {language} dataset from {dataset_file}")
        
        examples = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    example = self._parse_line(line.strip())
                    if example:
                        examples.append(example)
                        
                except Exception as e:
                    logger.warning(f"Error parsing line {i+1}: {e}")
                    continue
        
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(examples)
        
        logger.info(f"Successfully loaded {len(examples)} examples")
        return examples
    
    def _find_dataset_file(self, language: str) -> Path:
        """Find the dataset file for a given language."""
        # Try different naming patterns
        patterns = [
            f"{language}_pii_*.jsonl",
            f"{language}_*.jsonl",
            f"*{language}*.jsonl"
        ]
        
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                return files[0]  # Return first match
        
        # Fallback to exact name
        return self.data_dir / f"{language}_pii_43k.jsonl"
    
    def _parse_line(self, line: str) -> Optional[PIIExample]:
        """Parse a single line from the JSONL file."""
        if not line.strip():
            return None
            
        try:
            data = json.loads(line)
            
            # Parse privacy_mask (stored as string)
            privacy_mask = {}
            if data.get('privacy_mask'):
                try:
                    privacy_mask = eval(data['privacy_mask'])
                except:
                    privacy_mask = {}
            
            # Parse span_labels (stored as string)
            span_labels = []
            if data.get('span_labels'):
                try:
                    span_labels = eval(data['span_labels'])
                except:
                    span_labels = []
            
            return PIIExample(
                masked_text=data.get('masked_text', ''),
                unmasked_text=data.get('unmasked_text', ''),
                privacy_mask=privacy_mask,
                span_labels=span_labels,
                bio_labels=data.get('bio_labels', []),
                tokenised_text=data.get('tokenised_text', [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing line: {e}")
            return None
    
    def get_dataset_stats(self, examples: List[PIIExample]) -> Dict:
        """
        Calculate statistics for the loaded dataset.
        
        Args:
            examples: List of PIIExample objects
            
        Returns:
            Dictionary with dataset statistics
        """
        if not examples:
            return {}
        
        # Count entity types
        entity_counts = {}
        total_entities = 0
        text_lengths = []
        
        for example in examples:
            text_lengths.append(len(example.unmasked_text))
            
            for entity_type in example.get_entity_types():
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                total_entities += 1
        
        return {
            'total_examples': len(examples),
            'total_entities': total_entities,
            'unique_entity_types': len(entity_counts),
            'entity_type_counts': entity_counts,
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths)
        }
    
    def split_dataset(self, 
                     examples: List[PIIExample],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, List[PIIExample]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            examples: List of PIIExample objects
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            seed: Random seed for splitting
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        import random
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }