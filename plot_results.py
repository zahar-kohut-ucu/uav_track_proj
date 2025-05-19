import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

merged = pd.read_csv("merged_tracking_metrics.csv", index_col=0)
seqs = merged.index.tolist()

metrics = ['idf1', 'mota', 'precision', 'recall']

plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics, 1):
    bt_col = f"BT_{metric}"
    ds_col = f"DS_{metric}"
    
    plt.subplot(2, 2, i)
    x = range(len(seqs))
    width = 0.35
    plt.bar([xi - width/2 for xi in x], merged[bt_col], width=width, label='ByteTrack')
    plt.bar([xi + width/2 for xi in x], merged[ds_col], width=width, label='DeepSORT')
    plt.xticks(ticks=x, labels=seqs, rotation=45, ha='right')
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} per Sequence")
    plt.legend()

plt.suptitle("Tracking Metrics Comparison: ByteTrack vs DeepSORT", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("bar.png")

plt.figure(figsize=(8, 8))
for metric in metrics:
    sns.scatterplot(
        x=merged[f"BT_{metric}"],
        y=merged[f"DS_{metric}"],
        label=metric.upper()
    )

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel("ByteTrack")
plt.ylabel("DeepSORT")
plt.title("Scatter Comparison: ByteTrack vs DeepSORT")
plt.legend()
plt.grid(True)
plt.axis("scaled")
plt.savefig("scatter.png")

avg_bt = merged[[f"BT_{m}" for m in metrics]].mean()
avg_ds = merged[[f"DS_{m}" for m in metrics]].mean()

avg_summary = pd.DataFrame({
    "Metric": metrics,
    "ByteTrack": avg_bt.values,
    "DeepSORT": avg_ds.values
})
print(avg_summary)
