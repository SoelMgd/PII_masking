#!/usr/bin/env python3

"""
Test script to generate BERT training metrics visualization.
"""

import sys
import os
sys.path.append('experiments/visualization')

from results_visualization import bert_training_metrics

if __name__ == "__main__":
    print("Generating BERT training metrics plot...")
    
    # Create images directory if it doesn't exist
    os.makedirs("experiments/images", exist_ok=True)
    
    # Generate the plot
    bert_training_metrics(save_path="experiments/images")
    
    print("Done! Check experiments/images/bert_training_loss.png") 