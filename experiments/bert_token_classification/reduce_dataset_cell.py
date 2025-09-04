# =============================================================================
# ### CELL: REDUCE TRAINING DATASET SIZE
# =============================================================================

import random
from collections import Counter

def reduce_dataset_size(train_data, reduction_strategy="random", target_size=10000, 
                       min_examples_per_label=50, seed=42):
    """
    Reduce training dataset size with different strategies.
    
    Args:
        train_data: Dictionary with 'texts', 'token_ids', 'label_ids', etc.
        reduction_strategy: 'random', 'balanced', 'stratified'
        target_size: Target number of examples
        min_examples_per_label: Minimum examples per label (for balanced)
        seed: Random seed for reproducibility
    
    Returns:
        Reduced train_data dictionary
    """
    
    random.seed(seed)
    
    original_size = len(train_data['texts'])
    print(f"📊 Original dataset size: {original_size:,} examples")
    
    if target_size >= original_size:
        print("⚠️  Target size >= original size, no reduction needed")
        return train_data
    
    # Get all indices
    all_indices = list(range(original_size))
    
    if reduction_strategy == "random":
        # Simple random sampling
        selected_indices = random.sample(all_indices, target_size)
        print(f"🎲 Random sampling: {target_size:,} examples")
        
    elif reduction_strategy == "balanced":
        # Try to keep balanced representation of all labels
        print(f"⚖️  Balanced sampling with min {min_examples_per_label} per label...")
        
        # Count labels in dataset
        label_counts = Counter()
        label_to_indices = {}
        
        for idx in all_indices:
            # Get unique labels for this example (excluding O and padding)
            example_labels = [label for label in train_data['label_ids'][idx] 
                            if label != train_data['label_to_id'].get('O', -1) and label != -100]
            
            for label in set(example_labels):  # Unique labels only
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
                label_counts[label] += 1
        
        print(f"📈 Found {len(label_to_indices)} unique labels")
        
        # Select examples trying to maintain balance
        selected_indices = set()
        
        # First, ensure minimum examples per label
        for label, indices in label_to_indices.items():
            if len(indices) >= min_examples_per_label:
                selected_indices.update(random.sample(indices, min_examples_per_label))
            else:
                selected_indices.update(indices)  # Take all if less than minimum
        
        # Fill remaining slots randomly
        remaining_slots = target_size - len(selected_indices)
        if remaining_slots > 0:
            remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
            if remaining_indices:
                additional_indices = random.sample(
                    remaining_indices, 
                    min(remaining_slots, len(remaining_indices))
                )
                selected_indices.update(additional_indices)
        
        selected_indices = list(selected_indices)[:target_size]
        
    elif reduction_strategy == "stratified":
        # Stratified sampling based on text length and label diversity
        print(f"📊 Stratified sampling by text length and label diversity...")
        
        # Calculate features for stratification
        features = []
        for idx in all_indices:
            text_length = len(train_data['texts'][idx])
            unique_labels = len(set([label for label in train_data['label_ids'][idx] 
                                   if label != train_data['label_to_id'].get('O', -1) and label != -100]))
            features.append((idx, text_length, unique_labels))
        
        # Sort by features and take every nth example
        features.sort(key=lambda x: (x[1], x[2]))  # Sort by length, then label diversity
        step = len(features) // target_size
        selected_indices = [features[i][0] for i in range(0, len(features), max(1, step))][:target_size]
        
    else:
        raise ValueError(f"Unknown reduction strategy: {reduction_strategy}")
    
    # Create reduced dataset
    reduced_data = {}
    for key in train_data.keys():
        if key in ['texts', 'token_ids', 'label_ids']:
            reduced_data[key] = [train_data[key][idx] for idx in selected_indices]
        else:
            reduced_data[key] = train_data[key]  # Keep metadata unchanged
    
    print(f"✅ Reduced dataset size: {len(reduced_data['texts']):,} examples")
    print(f"📉 Reduction ratio: {len(reduced_data['texts'])/original_size:.1%}")
    
    # Show label distribution in reduced dataset
    if 'label_to_id' in train_data:
        reduced_label_counts = Counter()
        for label_seq in reduced_data['label_ids']:
            for label in label_seq:
                if label != train_data['label_to_id'].get('O', -1) and label != -100:
                    reduced_label_counts[label] += 1
        
        print(f"📊 Reduced dataset has {len(reduced_label_counts)} active labels")
        
        # Show top 10 most common labels
        if reduced_label_counts:
            id_to_label = train_data['id_to_label']
            print("🔝 Top 10 labels in reduced dataset:")
            for label_id, count in reduced_label_counts.most_common(10):
                label_name = id_to_label.get(label_id, f"ID_{label_id}")
                print(f"   {label_name}: {count:,} tokens")
    
    return reduced_data

# =============================================================================
# USAGE: Choose your reduction strategy and size
# =============================================================================

# Configuration for dataset reduction
REDUCTION_CONFIG = {
    'strategy': 'balanced',        # 'random', 'balanced', 'stratified'
    'target_size': 15000,         # Target number of examples (from ~94k)
    'min_examples_per_label': 20,  # Minimum examples per label (for balanced)
    'seed': 42                     # For reproducibility
}

print("🔄 Reducing training dataset size...")
print(f"📋 Strategy: {REDUCTION_CONFIG['strategy']}")
print(f"🎯 Target size: {REDUCTION_CONFIG['target_size']:,}")

# Apply reduction
train_data = reduce_dataset_size(
    train_data, 
    reduction_strategy=REDUCTION_CONFIG['strategy'],
    target_size=REDUCTION_CONFIG['target_size'],
    min_examples_per_label=REDUCTION_CONFIG['min_examples_per_label'],
    seed=REDUCTION_CONFIG['seed']
)

# Recreate the dataset and dataloader with reduced data
print("🔄 Recreating dataset and dataloader...")

train_dataset = FastPIITokenDataset(train_data, config.max_length)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True,
    num_workers=2
)

print(f"✅ New training dataset: {len(train_dataset):,} examples")
print(f"📊 New dataloader: {len(train_dataloader):,} batches")

# Update total training steps for scheduler
total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
print(f"📈 Updated total training steps: {total_steps:,}")

# Recreate scheduler with new steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=total_steps
)

print("🚀 Ready to train with reduced dataset!") 