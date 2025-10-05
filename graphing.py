# Re-import libraries after reset
import matplotlib.pyplot as plt

# Model data: names, sizes (MB), GLUE scores, GLUE errors
models = ["EmbBERT Nano", "EmbBERT tiny", "EmbBERT", "EmbBERT Med", "EmbBERT Big"]
## ----- FULL ------
# sizes_mb = [0.65, 1.24, 1.95, 9.60, 39.71]  # MB
# glue_scores = [55.26, 57.10, 63.5, 64.37, 65.53]
# glue_errors = [0.32, 0.71, 0.37, 0.48, 0.25]

## ----- QUANT ------
sizes_mb = [0.33, 0.51, 0.78, 3.12, 14.27]  # MB
glue_scores = [54.25, 57.82, 63.12, 62.13, 66.31]
glue_errors = [0.15, 0.12, 0.22, 0.17, 0.12]


# Baseline BERT-Tiny
bert_tiny_name = "BERT-Tiny"
bert_tiny_size = 20.0  # MB
bert_tiny_score = 63.16

plt.figure(figsize=(8, 6))
plt.errorbar(sizes_mb, glue_scores, yerr=glue_errors, fmt='none', ecolor='black', capsize=5)
plt.semilogx(sizes_mb, glue_scores, marker="o", color="red", linestyle="--", label="EmbBERT GLUE Score")

# Annotate points
for i, v in enumerate(glue_scores):
    plt.text(sizes_mb[i], v + 0.7, f"{v:.2f}%", ha="center", color="red")

# Add BERT-Tiny as a star
plt.plot(bert_tiny_size, bert_tiny_score, marker="*", color="gold", markersize=15, label="BERT-Tiny")
plt.text(bert_tiny_size, bert_tiny_score+0.2, f"{bert_tiny_score:.2f}%", ha="center", color="black")

plt.xticks(sizes_mb + [bert_tiny_size], models + [bert_tiny_name], rotation=15)
plt.xlabel("Model Size (MB) [log scale]")
plt.ylabel("GLUE Score (%)")
plt.title("GLUE Score vs Model Size (semilog) quant")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("GLUE_vs_size_quant.png", dpi=300)
plt.show()
