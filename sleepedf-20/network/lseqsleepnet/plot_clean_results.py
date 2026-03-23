import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Function to smooth curve
# -----------------------------
def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# -----------------------------
# Load training log
# -----------------------------
steps = []
train_acc = []
train_loss = []

with open("output/train_log.txt") as f:
    for line in f:
        parts = line.split()
        steps.append(int(float(parts[0])))
        train_loss.append(float(parts[2]))
        train_acc.append(float(parts[3]))

# Smooth training curves
smooth_acc = moving_average(train_acc, 200)
smooth_loss = moving_average(train_loss, 200)

# -----------------------------
# Plot Clean Training Accuracy
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(smooth_acc, color="blue")
plt.title("Smoothed Training Accuracy")
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.grid(True)
plt.savefig("output/clean_training_accuracy.png")
plt.close()

# -----------------------------
# Plot Clean Training Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(smooth_loss, color="red")
plt.title("Smoothed Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("output/clean_training_loss.png")
plt.close()

# -----------------------------
# Validation Accuracy
# -----------------------------
val_acc = []
with open("output/eval_result_log.txt") as f:
    for line in f:
        parts = line.split()
        val_acc.append(float(parts[-1]))

plt.figure(figsize=(8,5))
plt.plot(val_acc, marker="o")
plt.title("Validation Accuracy Over Time")
plt.xlabel("Validation Iteration")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.grid(True)
plt.savefig("output/clean_validation_accuracy.png")
plt.close()

print("Clean graphs saved in output/")
