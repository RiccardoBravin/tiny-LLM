import os
import re
import numpy as np

GLUE_TASKS = {
    "cola":      ("eval_mcc", "CoLA"),
    "sst2":      ("eval_accuracy", "SST-2"),
    "mrpc":      ("eval_f1", "MRPC"),
    "stsb":      ("eval_scc", "STSB"),
    "qqp":       ("eval_f1", "QQP"),
    "mnli-m":    ("eval_accuracy", "MNLI-m"),
    "mnli-mm":   ("eval_accuracy", "MNLI-mm"),
    "qnli":      ("eval_accuracy", "QNLI"),
    "rte":       ("eval_accuracy", "RTE"),
    "wnli":      ("eval_accuracy", "WNLI"),
}

OTHER_TASKS = [
    "imdb", "news", "bull", "limit", "emotion_split", "nlu", "snips"
]

def parse_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    blocks = content.strip().split("-------------------------------")
    metrics = []
    for block in blocks:
        if not block.strip():
            continue
        metric_dict = {}
        for line in block.strip().splitlines():
            m = re.match(r"(\w+):\s*([-\d\.eE]+)", line)
            if m:
                key, value = m.groups()
                metric_dict[key] = float(value)
        metrics.append(metric_dict)
    return metrics

def aggregate_metrics(metrics, key, reverse=False):
    vals = [m[key] for m in metrics if key in m]
    if reverse:
        vals = [1.0 - v for v in vals]
    if not vals:
        return None, None
    arr = np.array(vals)
    return arr.mean()*100, arr.std()*100

def main(results_dir):
    glue_scores = []
    glue_errors = []
    print("GLUE Results:")
    for fname, (metric_key, display_name) in GLUE_TASKS.items():
        fpath = os.path.join(results_dir, fname + ".txt")
        if not os.path.exists(fpath):
            print(f"  {display_name}: MISSING")
            continue
        metrics = parse_file(fpath)
        reverse = (fname == "wnli" and metric_key == "eval_accuracy")
        mean, std = aggregate_metrics(metrics, metric_key, reverse=reverse)
        glue_scores.append(mean)
        glue_errors.append(std)
        print(f"  {display_name:8}: {mean:.4f} ± {std:.4f} ({metric_key})")

    # Only average over non-None scores
    glue_scores_clean = [s for s in glue_scores if s is not None]
    glue_errors_clean = [e for e in glue_errors if e is not None]
    glue_score = np.mean(glue_scores_clean)
    glue_error = np.sqrt(np.sum(np.array(glue_errors_clean) ** 2)) / len(glue_errors_clean)
    print(f"\nGLUE SCORE: {glue_score:.4f} ± {glue_error:.4f}")

    print("\nOther Classification Results (Accuracy):")
    tinynlp_scores = []
    tinynlp_errors = []
    for fname in OTHER_TASKS:
        fpath = os.path.join(results_dir, fname + ".txt")
        if not os.path.exists(fpath):
            print(f"  {fname}: MISSING")
            continue
        metrics = parse_file(fpath)
        mean, std = aggregate_metrics(metrics, "eval_accuracy")
        tinynlp_scores.append(mean)
        tinynlp_errors.append(std)
        print(f"  {fname:15}: {mean:.4f} ± {std:.4f}")

    tinynlp_scores_clean = [s for s in tinynlp_scores if s is not None]
    tinynlp_errors_clean = [e for e in tinynlp_errors if e is not None]
    other_score = np.mean(tinynlp_scores_clean)
    other_error = np.sqrt(np.sum(np.array(tinynlp_errors_clean) ** 2)) / len(tinynlp_errors_clean)
    print(f"\nAVG ACCURACY (tinyNLP): {other_score:.4f} ± {other_error:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results.py <results_dir>")
    else:
        main(sys.argv[1])
