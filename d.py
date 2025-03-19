import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)

plt.figure(figsize=(8, 6))
sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="viridis", fill=True, thresh=0.05)

plt.title("2D Distribution Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()
