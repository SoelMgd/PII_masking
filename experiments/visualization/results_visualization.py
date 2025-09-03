import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def f1_score_class(json_file_path, save_path="experiments/images"):
    """
    Load a JSON result file and display a bar plot with the F1-score of each label.
    
    Args:
        json_file_path (str): Path to the JSON result file
        save_path (str): Directory to save the plot image
    """
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract per-class metrics
    per_class_metrics = data.get('per_class_metrics', {})
    
    # Extract class names and F1 scores
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[cls]['f1_score'] for cls in classes]
    
    # Sort by F1 score for better visualization
    sorted_data = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
    classes_sorted, f1_scores_sorted = zip(*sorted_data)
    
    # Create the bar plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(classes_sorted)), f1_scores_sorted, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('PII Classes', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'F1-Score by PII Class\n{Path(json_file_path).stem}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = f"{Path(json_file_path).stem}_f1_scores.png"
    save_file_path = os.path.join(save_path, filename)
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"F1-score plot saved to: {save_file_path}")
    plt.show()

def class_distribution(class_counts_file="data/pii_class_counts.json", save_path="experiments/images"):
    """
    Display a bar plot showing the distribution of PII classes in the dataset.
    
    Args:
        class_counts_file (str): Path to the JSON file containing class counts
        save_path (str): Directory to save the plot image
    """
    # Load the class counts file
    with open(class_counts_file, 'r') as f:
        data = json.load(f)
    
    # Extract class counts (first element of the list, excluding 'language' key)
    class_counts = data[0].copy()
    if 'language' in class_counts:
        del class_counts['language']
    
    # Extract class names and counts
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Sort by count for better visualization
    sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
    classes_sorted, counts_sorted = zip(*sorted_data)
    
    # Create the bar plot
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(classes_sorted)), counts_sorted, 
                   color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('PII Classes', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('Distribution of PII Classes in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_sorted)*0.01, 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "class_distribution.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to: {save_file_path}")
    plt.show()

def compare_models(json_file_paths, save_path="experiments/images"):
    """
    Take a list of JSON files and display a bar plot with global metrics 
    (precision, recall, f1_score) for each model.
    
    Args:
        json_file_paths (list): List of paths to JSON result files
        save_path (str): Directory to save the plot image
    """
    models = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Load data from each file
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract model name from filename
        model_name = Path(json_file_path).stem
        models.append(model_name)
        
        # Extract global metrics
        precisions.append(data.get('precision', 0))
        recalls.append(data.get('recall', 0))
        f1_scores.append(data.get('f1_score', 0))
    
    # Set up the bar plot
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(15, 8))
    
    # Create bars for each metric
    bars1 = plt.bar(x - width, precisions, width, label='Precision', 
                    color='lightcoral', alpha=0.8)
    bars2 = plt.bar(x, recalls, width, label='Recall', 
                    color='lightgreen', alpha=0.8)
    bars3 = plt.bar(x + width, f1_scores, width, label='F1-Score', 
                    color='lightskyblue', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison: Global Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x, [name.replace('_', '\n') for name in models], rotation=0, ha='center')
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "model_comparison.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {save_file_path}")
    plt.show()

def main():
    """
    Main function to execute the visualization functions for the specified models.
    """
    # Base path for results
    results_path = "results"
    
    # List of JSON files to process
    json_files = [
        f"{results_path}/mistral_finetuned.json",
        f"{results_path}/mistral_large_few_shot.json",
        f"{results_path}/mistral_medium_few_shot.json",
        f"{results_path}/mistral_small_few_shot.json"
    ]
    
    # Check which files exist
    existing_files = []
    for file_path in json_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"Found: {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not existing_files:
        print("No JSON files found. Please check the file paths.")
        return
    
    print("\n" + "="*50)
    print("Generating class distribution plot...")
    print("="*50)
    
    # Generate class distribution plot
    try:
        class_distribution()
    except Exception as e:
        print(f"Error generating class distribution plot: {e}")
    
    print("\n" + "="*50)
    print("Generating F1-score plots for each model...")
    print("="*50)
    
    # Generate F1-score plots for each model
    for json_file in existing_files:
        print(f"\nProcessing: {json_file}")
        try:
            f1_score_class(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("\n" + "="*50)
    print("Generating model comparison plot...")
    print("="*50)
    
    # Generate model comparison plot
    try:
        compare_models(existing_files)
    except Exception as e:
        print(f"Error generating model comparison: {e}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 