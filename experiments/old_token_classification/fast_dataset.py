#!/usr/bin/env python3
"""
Fast Dataset class for token classification training.

This version uses pre-computed token IDs from dataset_processing.py
to eliminate tokenization overhead during training.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

class FastPIITokenDataset(Dataset):
    """Ultra-fast PyTorch Dataset for PII token classification."""
    
    def __init__(self, dataset: Dict[str, Any], max_length: int = 96):
        """
        Initialize dataset with pre-computed token IDs.
        
        Args:
            dataset: Processed dataset dictionary with token_ids
            max_length: Maximum sequence length
        """
        self.texts = dataset['texts']
        self.tokens = dataset['tokens']
        self.token_ids = dataset['token_ids']  # ✅ Pre-computed token IDs
        self.labels = dataset['labels']
        self.label_ids = dataset['label_ids']
        self.label_to_id = dataset['label_to_id']
        self.id_to_label = dataset['id_to_label']
        self.max_length = max_length
        
        print(f"✅ FastPIITokenDataset initialized with {len(self.texts)} examples")
        print(f"📊 Max length: {max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single example - ultra-fast, no tokenization needed."""
        # Get pre-computed data
        token_ids = self.token_ids[idx]  # ✅ Already computed!
        label_ids = self.label_ids[idx]  # ✅ Already aligned!
        
        # Pad/truncate to max_length
        input_ids = self._pad_or_truncate(token_ids, self.max_length, pad_value=0)
        labels = self._pad_or_truncate(label_ids, self.max_length, pad_value=-100)
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _pad_or_truncate(self, sequence: List[int], max_length: int, pad_value: int) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_value] * (max_length - len(sequence))

# =============================================================================
# ### CELL FOR KAGGLE: REPLACE YOUR DATASET CLASS WITH THIS
# =============================================================================

print("""
🚀 COPY THIS TO YOUR KAGGLE NOTEBOOK:

Replace your PIITokenDataset class with this FastPIITokenDataset:

class FastPIITokenDataset(Dataset):
    def __init__(self, dataset: Dict[str, Any], max_length: int = 96):
        self.texts = dataset['texts']
        self.token_ids = dataset['token_ids']  # ✅ Pre-computed!
        self.label_ids = dataset['label_ids']  # ✅ Pre-aligned!
        self.label_to_id = dataset['label_to_id']
        self.id_to_label = dataset['id_to_label']
        self.max_length = max_length
        print(f"✅ FastPIITokenDataset: {len(self.texts)} examples, max_length={max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # ✅ NO tokenization - just padding!
        token_ids = self.token_ids[idx]
        label_ids = self.label_ids[idx]
        
        # Pad/truncate
        input_ids = self._pad_or_truncate(token_ids, self.max_length, 0)
        labels = self._pad_or_truncate(label_ids, self.max_length, -100)
        attention_mask = [1 if tid != 0 else 0 for tid in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _pad_or_truncate(self, seq, max_len, pad_val):
        return seq[:max_len] + [pad_val] * max(0, max_len - len(seq))

# Then use it like:
# train_dataset = FastPIITokenDataset(train_data, config.max_length)
""") 