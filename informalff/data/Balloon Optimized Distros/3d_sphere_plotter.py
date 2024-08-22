import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = sys.argv[1]

with open(filename, 'r') as f:
    to_plot = json.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_aspect('equal')

u, v = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 100),
                   np.linspace(0, 2 * np.pi, 100))
r = 0.9
x = r * np.cos(u)*np.sin(v)
y = r * np.sin(u)*np.sin(v)
z = r * np.cos(v)
ax.plot_wireframe(x, y, z, color="b", alpha=0.2)

ax.scatter(to_plot['X'], to_plot['Y'], to_plot['Z'], color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Grid in sphere')
ax.grid(True)
plt.show()