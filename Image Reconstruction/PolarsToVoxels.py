
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
R_max = 10


R = np.arange(0, R_max, voxel_length/2)
points = np.array([1, 1, 1, 1])
weight = 1

vect1 = [0, 0, 1]
vect2 = [0, 0.001, 1]

Theta_sizes = 2 * np.arcsin((voxel_length/2)/(R*np.sin(theta_c)))


for x in range(0, len(Theta_sizes)):
    if math.isnan(Theta_sizes[x]):
        Theta_sizes[x] = np.pi
    else:
        break
def lin(a):
    return np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//a) + 1))
thetas = np.apply_along_axis(lin, axis=0, arr=Theta_sizes)
print(thetas)

for r in R:
    Theta_size = 2 * np.arcsin((voxel_length/2)/(r*np.sin(theta_c)))
    if math.isnan(Theta_size):
        Theta_size = np.pi
    Theta = np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//Theta_size) + 1))
    for t in Theta:
        Rot = rotation_matrix(vect1, vect2)
        new = Rot.dot([r * np.sin(theta_c) * np.cos(t), r * np.sin(theta_c) * np.sin(t), np.cos(theta_c) * r])
        point = np.array([*list(new), weight])
        points = np.row_stack((points, point))

points = np.delete(points, 0, axis=0)

print(len(points))
# Plot the surface
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap=plt.cm.twilight_shifted)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 20)
plt.show()
