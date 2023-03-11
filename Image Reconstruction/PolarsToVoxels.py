
"""
Created on Thu Feb 16 14:43:24 2023
@author: amberjones, built upon by Joe Evans
"""
import math
import time
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def rotation_matrix(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

theta_c = (np.pi/8)-0.001
voxel_length = 1
R_max = 20


R = np.arange(0, R_max, voxel_length/2)
points = []
weight = 1

vect1 = [0, 0, 1]
vect2 = [0, 0.0001, 1]

Theta_sizes = 2 * np.arcsin((voxel_length/2)/(R*np.sin(theta_c)))

Rot = rotation_matrix(vect1, vect2)
counter = 0
for r in R:
    point_per_circle_in_area = False
    Theta_size = 2 * np.arcsin((voxel_length) / (r * np.sin(theta_c)))
    if math.isnan(Theta_size):
        Theta_size = np.pi
    Theta = np.linspace(0, 2 * np.pi, 2 * int(np.abs(2 * np.pi // Theta_size) + 1))
    for t in Theta:
        rotated_point = Rot.dot(
            [r * np.sin(theta_c) * np.cos(t), r * np.sin(theta_c) * np.sin(t), np.cos(theta_c) * r])
        point = [*list(rotated_point)]
        points.append(point)
        counter += 1
        point_per_circle_in_area = True
    if not point_per_circle_in_area:
        break
print(counter)
points = np.reshape(points, (counter, 3))
print(len(points))
print(points)
# Plot the surface
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blueviolet')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 20)
ax.axis()
plt.savefig('pointcone.png')
plt.show()
