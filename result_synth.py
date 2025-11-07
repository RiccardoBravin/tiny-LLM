import os
import re
import math

def extract_metrics(file_path):
    """Extract metrics from a result file."""
    with open(file_path, 'r') as f:
        content = f.read()

    metrics = {}
    patterns = {
        'accuracy': r'eval_accuracy:\s*(-?[\d.]+)',
        'f1': r'eval_f1:\s*(-?[\d.]+)',
        'mcc': r'eval_mcc:\s*(-?[\d.]+)',
        'scc': r'eval_scc:\s*(-?[\d.]+)'  # Assuming SCC might be in the file
    }
    for key, pattern in patterns.items():
        match = re.findall(pattern, content)
        if match:
            metrics[key] = list(map(float, match))
    return metrics

def calculate_averages_and_cis(metrics):
    """Calculate the average and confidence interval for each metric."""
    averages = {}
    cis = {}
    for key, values in metrics.items():
        if values:
            avg = sum(values) / len(values)
            std_dev = math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))
            ci = 1.96 * (std_dev / math.sqrt(len(values)))
            averages[key] = avg
            cis[key] = ci
        else:
            averages[key] = None  # Indicates no data for this metric
            cis[key] = None
    return averages, cis

def process_results_for_model(model, results_dir, excel_output=False):
    """Process results for a single model and generate a summary table with confidence intervals."""
    model_path = os.path.join(results_dir, model)
    if not os.path.exists(model_path):
        print(f"Directory not found for model: {model}")
        print("-" * 80 + "\n\n")
        return

    # Prepare rows for output
    rows = []
    header = "Dataset\t\tAcc\tF1\tMCC\tSCC\tCI Acc\tCI F1\tCI MCC\tCI SCC"

    for file_name in os.listdir(model_path):
        file_path = os.path.join(model_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            #remove .txt from filename
            file_name = file_name[:-4]
            metrics = {'accuracy': [], 'f1': [], 'mcc': [], 'scc': []}
            file_metrics = extract_metrics(file_path)
            for key in metrics:
                metrics[key].extend(file_metrics.get(key, []))

            averages, cis = calculate_averages_and_cis(metrics)

            # Convert values to percentages
            def format_percentage(value):
                return f"{value * 100:.2f}" if value is not None else "N/A"

            # Add a row for averages and confidence intervals
            avg_row = [
                file_name,
                format_percentage(averages["accuracy"]),
                format_percentage(averages["f1"]),
                format_percentage(averages["mcc"]),
                format_percentage(averages["scc"]),
            ]
            ci_row = [
                "",
                format_percentage(cis["accuracy"]),
                format_percentage(cis["f1"]),
                format_percentage(cis["mcc"]),
                format_percentage(cis["scc"]),
            ]
            rows.append(avg_row + ci_row[1:])

    if excel_output:
        # Excel-friendly format: Tab-separated
        print(f"Model: {model}\n")
        print(header)
        for row in rows:
            print("\t".join(row).replace(".", ","))
        print("\n\n")
    else:
        # Pretty printed output
        print(f"Model: {model}\n")
        print(f"{'Dataset':<20} {'Acc':<10} {'F1':<10} {'MCC':<10} {'SCC':<10}")
        print(f"{' ':<20} {'CI Acc':<10} {'CI F1':<10} {'CI MCC':<10} {'CI SCC':<10}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
            print(f"{'':<20} {row[5]:<10} {row[6]:<10} {row[7]:<10} {row[8]:<10}")
        print("\n\n")



EXCEL = True  # Set to True if you want to print the results in Excel format

models = ['EmbBERT']  # Replace with your model name
results_dir = './results/finetuning'  # Root directory containing model folders
# results_dir = './results/quantization'  # Root directory containing model folders
for model in models:
    process_results_for_model(model, results_dir, EXCEL)

