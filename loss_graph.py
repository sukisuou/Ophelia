import numpy as np
import matplotlib.pyplot as plt

# load history dict (supports both pretrain and standard training files)
try:
    history = np.load("pretrain_history.npy", allow_pickle=True).item()
except FileNotFoundError:
    history = np.load("training_history.npy", allow_pickle=True).item()

loss = history["loss"]
val_loss = history.get("val_loss", None)
epochs = range(1, len(loss) + 1)

# plot loss and val_loss
plt.figure(figsize=(7, 4))
plt.plot(epochs, loss, marker="o", linewidth=2, label="Training Loss")

if val_loss is not None:
    plt.plot(epochs, val_loss, marker="s", linewidth=2, linestyle="--", label="Validation Loss")

plt.title("Ophelia Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# save and display
plt.savefig("loss_curve.png")
plt.show()