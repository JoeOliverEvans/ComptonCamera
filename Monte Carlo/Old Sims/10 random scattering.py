# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:35:47 2023

@author: dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters of the simulation
source_energy = 511 # keV
detector_thickness = 0.5 # cm
detector_efficiency = 0.5 # 50%
detector_resolution = 0.05 # keV
n_photons = 10 # Number of photons to simulate

# Define the screen and detector positions
screen_pos = np.array([0, 10, 0])
detector_pos = np.array([0, 0, 10])
scatter_screen_dimensions = [25, 0.5, 25]
detector_dimensions = [25, 0.5, 25]

# Generate a random position for the source
source_position = np.random.rand(3) + 2*screen_pos[1]

# Create the figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the initial position and direction of the two photons
dir = []
for i in range(n_photons):
    x = np.random.random() * scatter_screen_dimensions[0]
    y = 0
    z = np.random.random() * scatter_screen_dimensions[2]
    pos1 = source_position
    dir.append(screen_pos - np.array([scatter_screen_dimensions[0]/2, 0,scatter_screen_dimensions[2]/2]) + np.array([x,y,z])- source_position)
    #dir[i] = screen_pos - np.array([scatter_screen_dimensions[0]/2, 0,scatter_screen_dimensions[2]/2]) + np.array([x,y,z])- source_position 
    # Calculate the intersection points with the screen and detector
    intersection1_screen = pos1 + np.dot((screen_pos - pos1), dir[i]) / np.dot(dir[i], dir[i]) * dir[i]
    intersection1_detector = intersection1_screen + np.dot((detector_pos - intersection1_screen), dir[i]) / np.dot(dir[i], dir[i]) * dir[i]
    # Plot the photons' path
    ax.plot([pos1[0], intersection1_screen[0], intersection1_detector[0]],
        [pos1[1], intersection1_screen[1], intersection1_detector[1]],
        [pos1[2], intersection1_screen[2], intersection1_detector[2]], 'r')
    
# Change the path of the photons as they scatter
E_0 = source_energy
photon = []
for i in range(n_photons):
    E = np.random.random() * E_0
    photon.append([dir[i], E])
     


# Plot the detector position
detector_position = np.array([0, 0, 0])
x = np.linspace(detector_position[0] - detector_dimensions[0]/2, detector_position[0] + detector_dimensions[0]/2, 10)
z = np.linspace(detector_position[2] - detector_dimensions[2]/2, detector_position[2] + detector_dimensions[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), detector_position[1])
ax.plot_surface(X, Y, Z)

# Plot the scattering material position
x = np.linspace(screen_pos[0] - scatter_screen_dimensions[0]/2, screen_pos[0] + scatter_screen_dimensions[0]/2, 10)
z = np.linspace(screen_pos[2] - scatter_screen_dimensions[2]/2, screen_pos[2] + scatter_screen_dimensions[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), screen_pos[1])
ax.plot_surface(X, Y, Z)

# Plot the source position
ax.scatter(source_position[0], source_position[1], source_position[2], color='green', label='Source')

# Show the plot
plt.show()
