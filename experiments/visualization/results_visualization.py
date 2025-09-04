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
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    per_class_metrics = data.get('per_class_metrics', {})
    
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[cls]['f1_score'] for cls in classes]
    
    sorted_data = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
    classes_sorted, f1_scores_sorted = zip(*sorted_data)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(classes_sorted)), f1_scores_sorted, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('PII Classes', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'F1-Score by PII Class\n{Path(json_file_path).stem}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
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
    with open(class_counts_file, 'r') as f:
        data = json.load(f)
    
    class_counts = data[0].copy()
    if 'language' in class_counts:
        del class_counts['language']
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
    classes_sorted, counts_sorted = zip(*sorted_data)
    
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(classes_sorted)), counts_sorted, 
                   color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    
    plt.xlabel('PII Classes', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('Distribution of PII Classes in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
    
    for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_sorted)*0.01, 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
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
    
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        model_name = Path(json_file_path).stem
        models.append(model_name)
        
        precisions.append(data.get('precision', 0))
        recalls.append(data.get('recall', 0))
        f1_scores.append(data.get('f1_score', 0))
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(15, 8))
    
    bars1 = plt.bar(x - width, precisions, width, label='Precision', 
                    color='lightcoral', alpha=0.8)
    bars2 = plt.bar(x, recalls, width, label='Recall', 
                    color='lightgreen', alpha=0.8)
    bars3 = plt.bar(x + width, f1_scores, width, label='F1-Score', 
                    color='lightskyblue', alpha=0.8)
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison: Global Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x, [name.replace('_', '\n') for name in models], rotation=0, ha='center')
    plt.ylim(0, 1.1)
    plt.legend()
    
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
    
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "model_comparison.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {save_file_path}")
    plt.show()

def bert_distilled_training_metrics(save_path="experiments/images"):
    """
    Plot DistilBERT training metrics: Training Loss and F1-Score over training steps.
    Uses data extracted from the DistilBERT fine-tuning notebook.
    
    Args:
        save_path (str): Directory to save the plot image
    """
    steps = [500, 1000, 1500, 2000, 2500, 3000]
    
    training_loss = [0.3731, 0.1520, 0.1136, 0.0964, 0.0910, 0.0809]
    
    validation_loss = [0.3026, 0.1337, 0.1043, 0.0894, 0.0832, 0.0813]
    
    f1_scores = [0.5894, 0.8402, 0.8866, 0.9035, 0.9093, 0.9131]
    
    precision = [0.6672, 0.8710, 0.8957, 0.9098, 0.9182, 0.9184]
    
    recall = [0.6076, 0.8446, 0.8950, 0.9131, 0.9137, 0.9191]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', color=color1, fontsize=12)
    line1 = ax1.plot(steps, training_loss, color=color1, marker='o', linewidth=2, 
                     label='Training Loss', markersize=6)
    line2 = ax1.plot(steps, validation_loss, color='tab:orange', marker='s', linewidth=2, 
                     label='Validation Loss', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('F1-Score', color=color2, fontsize=12)
    line3 = ax2.plot(steps, f1_scores, color=color2, marker='^', linewidth=2, 
                     label='F1-Score (Token-level)', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)
    
    plt.title('DistilBERT Fine-tuning: Training Progress\n(DistilBERT on PII Token Classification)', 
              fontsize=14, fontweight='bold', pad=20)
    
    for i, (step, loss, f1) in enumerate(zip(steps, training_loss, f1_scores)):
        if i % 2 == 0:
            ax1.annotate(f'{loss:.3f}', (step, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=color1)
            ax2.annotate(f'{f1:.3f}', (step, f1), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.9)
    
    textstr = f'Final Metrics (Step 3000):\n• F1-Score: {f1_scores[-1]:.3f}\n• Precision: {precision[-1]:.3f}\n• Recall: {recall[-1]:.3f}\n• Val Loss: {validation_loss[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "bert_distilled_training_loss.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"DistilBERT training metrics plot saved to: {save_file_path}")
    plt.show()

def bert_classic_training_metrics(save_path="experiments/images"):
    """
    Plot BERT Classic training metrics: Training Loss and F1-Score over training steps.
    Uses data from BERT Classic fine-tuning results.
    
    Args:
        save_path (str): Directory to save the plot image
    """
    steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    training_loss = [0.2546, 0.1034, 0.0889, 0.0785, 0.0744, 0.0606, 0.0619, 0.0581]
    
    validation_loss = [0.2024, 0.1011, 0.0859, 0.0780, 0.0725, 0.0732, 0.0718, 0.0703]
    
    f1_scores = [0.6888, 0.8873, 0.9040, 0.9233, 0.9195, 0.9320, 0.9260, 0.9354]
    
    precision = [0.7441, 0.8767, 0.8953, 0.9220, 0.9215, 0.9313, 0.9306, 0.9342]
    
    recall = [0.7074, 0.9036, 0.9199, 0.9329, 0.9303, 0.9384, 0.9324, 0.9421]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', color=color1, fontsize=12)
    line1 = ax1.plot(steps, training_loss, color=color1, marker='o', linewidth=2, 
                     label='Training Loss', markersize=6)
    line2 = ax1.plot(steps, validation_loss, color='tab:orange', marker='s', linewidth=2, 
                     label='Validation Loss', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('F1-Score', color=color2, fontsize=12)
    line3 = ax2.plot(steps, f1_scores, color=color2, marker='^', linewidth=2, 
                     label='F1-Score (Token-level)', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)
    
    plt.title('BERT Classic Fine-tuning: Training Progress\n(BERT-base on PII Token Classification)', 
              fontsize=14, fontweight='bold', pad=20)
    
    for i, (step, loss, f1) in enumerate(zip(steps, training_loss, f1_scores)):
        if i % 2 == 0:
            ax1.annotate(f'{loss:.3f}', (step, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=color1)
            ax2.annotate(f'{f1:.3f}', (step, f1), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.9)
    
    textstr = f'Final Metrics (Step 4000):\n• F1-Score: {f1_scores[-1]:.3f}\n• Precision: {precision[-1]:.3f}\n• Recall: {recall[-1]:.3f}\n• Val Loss: {validation_loss[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "bert_classic_training_loss.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"BERT Classic training metrics plot saved to: {save_file_path}")
    plt.show()

def bert_classic_f1_by_class(json_file_path=None, save_path="experiments/images"):
    """
    Create and save a bar plot showing F1-scores by class for BERT Classic model.
    
    Args:
        json_file_path (str): Path to the JSON result file containing BERT Classic results
        save_path (str): Directory to save the plot image
    """
    if json_file_path is None:
        print("Error: Please provide the path to BERT Classic evaluation results JSON file")
        print("Usage: bert_classic_f1_by_class('path/to/bert_classic_results.json')")
        return None
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return None
    
    # Load data from JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    per_class_metrics = data.get('per_class_metrics', {})
    
    if not per_class_metrics:
        print("Error: No per_class_metrics found in the JSON file")
        return None
    
    classes = list(per_class_metrics.keys())
    f1_scores = [per_class_metrics[cls]['f1_score'] for cls in classes]
    
    # Sort by F1-score (descending)
    sorted_data = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
    classes_sorted, f1_scores_sorted = zip(*sorted_data)
    
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(classes_sorted)), f1_scores_sorted, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('PII Classes', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('BERT Classic: F1-Score by PII Class\n(Entity-level Evaluation)', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores_sorted)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "bert_classic_f1_by_class.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"BERT Classic F1-scores by class plot saved to: {save_file_path}")
    plt.show()
    
    return save_file_path

def compare_bert_mistral_models(save_path="experiments/images"):
    """
    Compare BERT models with Mistral models using entity-level metrics.
    Creates a bar plot comparing F1-scores, Precision, and Recall.
    
    Args:
        save_path (str): Directory to save the plot image
    """
    models = ['Mistral\nFine-tuned', 'BERT Classic\n(Entity-level)', 'Mistral Large\n(Few-shot)', 'DistilBERT\n(Entity-level)']
    
    f1_scores = [0.892, 0.7139, 0.647, 0.61]  # Mistral FT, BERT Classic, Mistral Large FS, DistilBERT
    precisions = [0.901, 0.6654, 0.689, 0.642]  # Mistral FT, BERT Classic, Mistral Large FS, DistilBERT  
    recalls = [0.883, 0.7700, 0.608, 0.580]  # Mistral FT, BERT Classic, Mistral Large FS, DistilBERT
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    
    bars1 = plt.bar(x - width, precisions, width, label='Precision', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    bars2 = plt.bar(x, recalls, width, label='Recall', 
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    bars3 = plt.bar(x + width, f1_scores, width, label='F1-Score', 
                    color='lightskyblue', alpha=0.8, edgecolor='darkblue')
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Entity-Level Performance Comparison\nMistral vs BERT Models', fontsize=14, fontweight='bold')
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=11)
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, "bert_mistral_entity_comparison.png")
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    print(f"BERT vs Mistral comparison plot saved to: {save_file_path}")
    plt.show()

def main():
    """
    Main function to execute the visualization functions for the specified models.
    """
    results_path = "results"
    
    json_files = [
        f"{results_path}/mistral_finetuned.json",
        f"{results_path}/mistral_large_few_shot.json",
        f"{results_path}/mistral_medium_few_shot.json",
        f"{results_path}/mistral_small_few_shot.json"
    ]
    
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
    
    try:
        class_distribution()
    except Exception as e:
        print(f"Error generating class distribution plot: {e}")
    
    print("\n" + "="*50)
    print("Generating F1-score plots for each model...")
    print("="*50)
    
    for json_file in existing_files:
        print(f"\nProcessing: {json_file}")
        try:
            f1_score_class(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("\n" + "="*50)
    print("Generating model comparison plot...")
    print("="*50)
    
    try:
        compare_models(existing_files)
    except Exception as e:
        print(f"Error generating model comparison: {e}")
    
    print("\n" + "="*50)
    print("Generating BERT training metrics plots...")
    print("="*50)
    
    # Generate DistilBERT training metrics plot
    try:
        bert_distilled_training_metrics()
    except Exception as e:
        print(f"Error generating DistilBERT training metrics: {e}")
    
    # Generate BERT Classic training metrics plot
    try:
        bert_classic_training_metrics()
    except Exception as e:
        print(f"Error generating BERT Classic training metrics: {e}")
    
    print("\n" + "="*50)
    print("Generating BERT Classic F1-scores by class plot...")
    print("="*50)
    
    # Generate BERT Classic F1-scores by class plot
    # You need to provide the path to your BERT Classic evaluation results JSON file
    bert_classic_json = "results/bert_classic_results.json"  # Update this path
    
    if os.path.exists(bert_classic_json):
        try:
            bert_classic_f1_by_class(bert_classic_json)
        except Exception as e:
            print(f"Error generating BERT Classic F1-scores by class: {e}")
    else:
        print(f"BERT Classic results file not found: {bert_classic_json}")
        print("Please update the path in the main() function or provide the JSON file")
    
    print("\n" + "="*50)
    print("Generating BERT vs Mistral comparison plot...")
    print("="*50)
    
    # Generate BERT vs Mistral comparison plot
    try:
        compare_bert_mistral_models()
    except Exception as e:
        print(f"Error generating BERT vs Mistral comparison: {e}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    import sys
    
    # Allow calling specific functions from command line
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        
        if function_name == "bert_classic_f1":
            if len(sys.argv) > 2:
                json_path = sys.argv[2]
                print(f"Generating BERT Classic F1-scores by class plot from {json_path}...")
                bert_classic_f1_by_class(json_path)
            else:
                print("Error: Please provide the JSON file path")
                print("Usage: python results_visualization.py bert_classic_f1 path/to/bert_results.json")
        elif function_name == "bert_classic_training":
            print("Generating BERT Classic training metrics plot...")
            bert_classic_training_metrics()
        elif function_name == "bert_distilled_training":
            print("Generating DistilBERT training metrics plot...")
            bert_distilled_training_metrics()
        elif function_name == "compare_models":
            print("Generating model comparison plot...")
            compare_bert_mistral_models()
        else:
            print(f"Unknown function: {function_name}")
            print("Available functions: bert_classic_f1, bert_classic_training, bert_distilled_training, compare_models")
    else:
        main() 