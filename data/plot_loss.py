import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

df = pd.read_csv("loss_landscape.csv")

resolution = int(np.sqrt(len(df)))
alpha = df["alpha"].values.reshape(resolution, resolution)
beta  = df["beta"].values.reshape(resolution, resolution)
loss  = df["loss"].values.reshape(resolution, resolution)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3D surface
ax3d = fig.add_subplot(121, projection="3d")
ax3d.plot_surface(alpha, beta, loss, cmap=cm.viridis, linewidth=0, antialiased=True)
ax3d.set_xlabel("Direction 1 (alpha)")
ax3d.set_ylabel("Direction 2 (beta)")
ax3d.set_zlabel("MSE Loss")
ax3d.set_title("Loss Landscape - 3D Surface")

# Right: contour heatmap
ax2d = axes[1]
contour_filled = ax2d.contourf(alpha, beta, loss, levels=40, cmap=cm.viridis)
contour_lines  = ax2d.contour(alpha, beta, loss, levels=10, colors="white", linewidths=0.5, alpha=0.4)
fig.colorbar(contour_filled, ax=ax2d, label="MSE Loss")
ax2d.set_xlabel("Direction 1 (alpha)")
ax2d.set_ylabel("Direction 2 (beta)")
ax2d.set_title("Loss Landscape - Contour")

# Mark the trained minimum (alpha=0, beta=0)
ax2d.plot(0, 0, "r*", markersize=12, label="Trained weights")
ax2d.legend()

plt.tight_layout()
plt.savefig("loss_landscape.png", dpi=150)
plt.show()