import os
import re
from pathlib import Path

def parse_metrics_file(filepath):
    """Parse a single results file and extract all metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all metric blocks
    accuracy_values = re.findall(r'Accuracy:\s+([\d.]+)', content)
    f1_values = re.findall(r'F1:\s+([\d.]+)', content)
    mcc_values = re.findall(r'MCC:\s+([\d.]+)', content)
    
    return {
        'accuracy': [float(x) for x in accuracy_values],
        'f1': [float(x) for x in f1_values],
        'mcc': [float(x) for x in mcc_values]
    }

def calculate_file_averages(folder_path):
    """Calculate average metrics for each file in folder."""
    folder = Path(folder_path)
    
    # Process each file
    for file_path in sorted(folder.glob('*.txt')):
        print(f"\n{'='*60}")
        print(f"File: {file_path.name}")
        print('='*60)
        
        metrics = parse_metrics_file(file_path)
        
        if metrics['accuracy']:
            avg_accuracy = sum(metrics['accuracy']) / len(metrics['accuracy'])
            avg_f1 = sum(metrics['f1']) / len(metrics['f1'])
            avg_mcc = sum(metrics['mcc']) / len(metrics['mcc'])
            
            # print(f"Runs found: {len(metrics['accuracy'])}")
            print(f"Average Accuracy: {(avg_accuracy*100):.4f}".replace('.', ','))
            print(f"Average F1:       {(avg_f1*100):.4f}".replace('.', ','))
            print(f"Average MCC:      {(avg_mcc*100):.4f}".replace('.', ','))
        else:
            print("No metrics found in this file!")

if __name__ == "__main__":
    # Change this to your folder path
    folder_path = "old results/results/mlm_NanoBERT_original_29"
    calculate_file_averages(folder_path)
