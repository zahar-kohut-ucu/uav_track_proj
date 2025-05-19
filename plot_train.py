import pandas as pd
import matplotlib.pyplot as plt

# Load results.csv
df = pd.read_csv("runs/detect/ft_visdrone_6/results.csv")

# --- Plot 1: Box loss train/val ---
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss (Train)', color='tab:blue')
plt.plot(df['epoch'], df['val/box_loss'], label='Box Loss (Val)', color='tab:orange')
plt.title('Box Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('box_loss_plot.png')
plt.close()

# --- Plot 2: mAP50 and mAP50-95 ---
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='tab:green')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='tab:red')
plt.title('Validation mAP Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('map_plot.png')
plt.close()
