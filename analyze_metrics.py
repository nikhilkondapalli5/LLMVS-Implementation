import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval_metrics.csv")

df = df.sort_values("video_id")

# We exclude 'video_id' from numeric metrics
metric_cols = [c for c in df.columns if c != "video_id"]

summary = df[metric_cols].describe().transpose()[["mean", "std", "min", "max"]]
print("===== Summary statistics =====")
print(summary.round(4))

summary.round(4).to_csv("metrics_summary.csv")
print("\nSummary saved to metrics_summary.csv")

per_video_metrics = ["rouge1_F1", "rouge2_F1", "rougeL_F1", "bertscore_F1", "cosine_sim"]

plt.figure(figsize=(10, 6))
for col in per_video_metrics:
    plt.plot(df["video_id"], df[col], marker="o", label=col)

plt.xlabel("Video ID")
plt.ylabel("Score")
plt.title("Per-video metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("per_video_metrics.png", dpi=200)
print("Per-video metrics plot saved to per_video_metrics.png")
plt.close()

means = summary["mean"]

plt.figure(figsize=(10, 6))
plt.bar(means.index, means.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average score")
plt.title("Average scores per metric")
plt.tight_layout()
plt.savefig("average_metrics.png", dpi=200)
print("Average metrics plot saved to average_metrics.png")
plt.close()