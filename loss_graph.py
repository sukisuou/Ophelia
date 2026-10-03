import numpy as np
import matplotlib.pyplot as plt

# load history dict
history = np.load("training_history.npy", allow_pickle = True).item()
loss = history["loss"]
epochs = range(1, len(loss) + 1)

# plot loss
plt.figure(figsize=(7, 4))
plt.plot(epochs, loss, marker="o", linewidth=2, label="Training Loss")
plt.title("Ophelia Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# save or display
plt.savefig("loss_curve.png")
plt.show()