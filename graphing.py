# Re-import libraries after reset
import matplotlib.pyplot as plt

# Model data: names, sizes (MB), GLUE scores, GLUE errors
models = ["EmbBERT Nano", "EmbBERT tiny", "EmbBERT", "EmbBERT Med", "EmbBERT Big"]
## ----- FULL ------
sizes_mb = [0.65, 1.24, 1.95, 9.60, 39.71]  # MB
glue_scores = [55.26, 57.10, 63.5, 64.37, 65.53]
glue_errors = [0.32, 0.71, 0.37, 0.48, 0.25]

## ----- QUANT ------
# sizes_mb = [0.33, 0.51, 0.78, 3.12, 14.27]  # MB
# glue_scores = [54.25, 57.82, 63.12, 62.13, 66.31]
# glue_errors = [0.15, 0.12, 0.22, 0.17, 0.12]


# Baseline BERT-Tiny
bert_tiny_name = "BERT-Tiny"
bert_tiny_size = 20.0  # MB
bert_tiny_score = 63.16

plt.figure(figsize=(8, 6))

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16
})

plt.errorbar(sizes_mb, glue_scores, yerr=glue_errors, fmt='none', ecolor='teal', capsize=5, elinewidth=2)
plt.semilogx(sizes_mb, glue_scores, marker="o", color="teal", linestyle="--", label="EmbBERT", markersize=10, linewidth=2)

# Annotate points
for i, v in enumerate(glue_scores):
    y_offset = 1.0 if i == 2 else -1.4
    plt.text(sizes_mb[i], v + y_offset, f"{v:.2f}%", ha="center", color="black")

# Add BERT-Tiny as a star
plt.plot(bert_tiny_size, bert_tiny_score, marker="*", color="gold", markersize=15, label="BERT-Tiny")
plt.text(bert_tiny_size, bert_tiny_score-1, f"{bert_tiny_score:.2f}%", ha="center", color="black")

plt.xticks(sizes_mb + [bert_tiny_size], models + [bert_tiny_name], rotation=20)
plt.xlabel("Model Size (MB) [log scale]", fontsize=14)
plt.ylabel("GLUE Score (%)", fontsize=14)
plt.title("GLUE Score vs Model Size (semilog)")
plt.legend(loc="lower right")
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Add left/right/top/bottom clearance
x_min = min(sizes_mb + [bert_tiny_size]) * 0.7
x_max = max(sizes_mb + [bert_tiny_size]) * 1.4
y_min = min(glue_scores + [bert_tiny_score]) - 2
y_max = max(glue_scores + [bert_tiny_score]) + 1
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.savefig("GLUE_vs_size_fp32.pdf")
plt.show()
