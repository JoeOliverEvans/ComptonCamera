"""
Created on Thu Feb 16 14:43:24 2023

@author: amberjones
"""

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

theta_c = 0.65
# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(-1, 1, 50)
theta = np.linspace(0, 2*np.pi, 50)
R, Theta = np.meshgrid(r, theta)
Z = (1/np.tan(theta_c))*R
# Express the mesh in the cartesian system.
X, Y = R*np.cos(Theta), R*np.sin(Theta)

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_zlim(-1, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')

plt.show()

