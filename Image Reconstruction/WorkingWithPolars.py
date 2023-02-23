"""
Created on Thu Feb 16 14:43:24 2023

@author: amberjones
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

N = 50
theta_c = 0.65

# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(0, 1, N)
theta = np.linspace(0, 2*np.pi, N)
R, Theta = np.meshgrid(r, theta)
Z = (1/np.tan(theta_c))*R
# Express the mesh in the cartesian system
X, Y = R*np.cos(Theta), R*np.sin(Theta)
# Plot the surface
ax.plot_surface(X, Y, Z, cmap=plt.cm.twilight_shifted)

ax.set_zlim(0, 1)
plt.show()
