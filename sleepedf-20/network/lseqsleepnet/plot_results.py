import matplotlib.pyplot as plt

# ---- Training Accuracy ----
steps = []
train_acc = []
train_loss = []

with open("output/train_log.txt") as f:
    for line in f:
        parts = line.split()
        steps.append(int(float(parts[0])))
        train_loss.append(float(parts[2]))
        train_acc.append(float(parts[3]))

plt.figure()
plt.plot(steps, train_acc)
plt.xlabel("Training Steps")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Curve")
plt.savefig("output/training_accuracy.png")
plt.close()


# ---- Training Loss ----
plt.figure()
plt.plot(steps, train_loss)
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.savefig("output/training_loss.png")
plt.close()


# ---- Validation Accuracy ----
val_acc = []

with open("output/eval_result_log.txt") as f:
    for line in f:
        parts = line.split()
        val_acc.append(float(parts[-1]))

plt.figure()
plt.plot(val_acc)
plt.xlabel("Validation Iteration")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Curve")
plt.savefig("output/validation_accuracy.png")
plt.close()

print("Plots saved inside output/ folder")
