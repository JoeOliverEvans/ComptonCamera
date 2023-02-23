
"""
Created on Thu Feb 16 14:43:24 2023
@author: amberjones, built upon by Joe Evans
"""
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

N = 50
theta_c = (np.pi/4)-0.001
voxel_length = 1
R_max = 4
# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(0, 1, N)

theta = np.linspace(0, 2*np.pi, N)

R = np.arange(0, R_max, voxel_length/4)
X = []
Y = []
Z = []
for r in R:
    Theta_size = 2 * np.arctan((voxel_length/4)/(np.sin(theta_c) * r))
    print(Theta_size)
    Theta = np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//Theta_size) + 1))
    for x in r*np.sin(Theta):
        X.append(x)
    for y in r*np.cos(Theta):
        Y.append(y)
        Z.append(np.cos(theta_c)*r)
print(X)
print(np.shape(X))
print(Y)
print(np.shape(Y))
print(Z)
print(np.shape(Z))


# Plot the surface
ax.scatter(X, Y, Z, c=Z, cmap=plt.cm.twilight_shifted)

ax.set_zlim(0, 2)
plt.show()
