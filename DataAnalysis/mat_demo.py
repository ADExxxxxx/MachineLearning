import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
noise = np.random.normal(4, 10)

x1 = np.random.normal(size=5).reshape(1, 5) + noise
x2 = np.random.normal(size=5).reshape(1, 5) + noise
x1, x2 = np.meshgrid(x1, x2)
colors = ['r', 'g', 'y', 'b', 'purple']
markers = ['v', 'o', 'x', 'p']
z = []
for i in range(10):
    z.append(np.random.rand(4, 10) * x1 + np.random.rand(4, 10) * x2 + np.random.rand(1, 10))

ax = plt.figure().add_subplot(111, projection='3d')
for i in range(10):
    ax.scatter(x1, x2, z[i - 1], color=colors[i % 5], marker=markers[i % 4])
plt.show()
