
"""
Created on Thu Feb 16 14:43:24 2023
@author: amberjones, built upon by Joe Evans
"""
import math

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

theta_c = (np.pi/40)-0.001
voxel_length = 1
R_max = 60


R = np.arange(0, R_max, voxel_length/2)
print(max(R))
points = np.array([1, 1, 1, 1])
weight = 1
for r in R:
    Theta_size = 2 * np.arcsin((voxel_length/2)/(r*np.sin(theta_c)))
    if math.isnan(Theta_size):
        Theta_size = np.pi
    Theta = np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//Theta_size) + 1))
    for t in Theta:
        point = np.array([r * np.sin(theta_c) * np.cos(t), r * np.sin(theta_c) * np.sin(t), np.cos(theta_c) * r, weight])
        points = np.row_stack((points, point))

points = np.delete(points, 0, axis=0)

# Plot the surface
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap=plt.cm.twilight_shifted)

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 40)
plt.show()
