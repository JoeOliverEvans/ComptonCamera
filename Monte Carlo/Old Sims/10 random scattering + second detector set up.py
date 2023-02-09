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
screen_pos = np.array([0, 10, 20])
detector_pos = np.array([0, 0, 20])
scatter_screen_dimensions = [50, 0.5, 50]
detector_dimensions = [50, 0.5, 50]
scr2_pos = np.array([50, 20, 20])
scr2_dim = [0.5, 50, 50]
det2_pos = np.array([65, 20, 20])
det2_dim = [0.5, 50, 50]
print(screen_pos[1])

#idk if this is correct orientation/positioning for source cuz visual looks weird

# Generate a random position for the source
source_position = np.random.rand(3) + 2*screen_pos[1]

# Create the figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the initial position and direction of the photons
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
    
# Define the initial position and direction of the photons towards second screen
dir2 = []
for i in range(n_photons):
    x = 0
    y = np.random.random() * scr2_dim[1]
    z = np.random.random() * scr2_dim[2]
    pos1 = source_position
    dir2.append(scr2_pos - np.array([scr2_dim[0]/2, 0,scr2_dim[2]/2]) + np.array([x,y,z])- source_position)
    #dir[i] = screen_pos - np.array([scatter_screen_dimensions[0]/2, 0,scatter_screen_dimensions[2]/2]) + np.array([x,y,z])- source_position 
    # Calculate the intersection points with the screen and detector
    intersection2_screen = pos1 + np.dot((scr2_pos - pos1), dir2[i]) / np.dot(dir2[i], dir2[i]) * dir2[i]
    intersection2_detector = intersection2_screen + np.dot((det2_pos - intersection2_screen), dir2[i]) / np.dot(dir2[i], dir2[i]) * dir2[i]
    print(intersection2_screen)
    
    # Plot the photons' path
    ax.plot([pos1[0], intersection2_screen[0], intersection2_detector[0]],
        [pos1[1], intersection2_screen[1], intersection2_detector[1]],
        [pos1[2], intersection2_screen[2], intersection2_detector[2]], 'r')    

def compton_scattering_angle(E_0, E):
    m_e = 5.11e-10 # Electron mass in keV/c^2


    # Calculate the scattering angle
    cos_theta = 1 - (E_0 / (E_0 + m_e)) * (1 - (E / (E + m_e)))
    theta = np.arccos(cos_theta)

    return theta

#code for rotational vector in x-z plane cuz this is whats happening in second screen?
"""
def rotate_vector_2d(v, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    v_rotated = np.dot(R, v)
    return v_rotated
"""
def rotate_vector_3d(v, axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    rotation_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    v_rotated = np.dot(rotation_matrix, v)
    return v_rotated

    
# Change the path of the photons as they scatter
E_0 = source_energy
photon = []
for i in range(n_photons):
    E = np.random.random() * E_0
    photon.append([dir[i], E])
    
# Change the path of the photons as they scatter towards second screen
E_0 = source_energy*1.6e-16
photon = []
for i in range(n_photons):
    E = np.random.random() * E_0
    photon.append([dir2[i], E])
    #m_e = 9.1e-31
    #c = 3e8
    #theta = np.arccos(1+((m_e * c**2)*((1/E_0 - 1/E))))
    #print(compton_scattering_angle(E_0, E))
    v = dir2[0] + dir2[2]
    #angle = compton_scattering_angle(E_0, E)
    #print(rotate_vector_2d(v, compton_scattering_angle(E_0, E)))
    pho_dir = dir2[i] # vector describing photon path
    #print(pho_dir)
    
    
    
    

     


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


# Plotting 2nd detector 
x = np.linspace(det2_pos[0] - det2_dim[0]/2, det2_pos[0] + det2_dim[0]/2, 10)
z = np.linspace(det2_pos[2] - det2_dim[2]/2, det2_pos[2] + det2_dim[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), det2_pos[1])
ax.plot_surface(X, Y, Z)

# Plotting 2nd detector 
x = np.linspace(scr2_pos[0] - scr2_dim[0]/2, scr2_pos[0] + scr2_dim[0]/2, 10)
z = np.linspace(scr2_pos[2] - scr2_dim[2]/2, scr2_pos[2] + scr2_dim[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), scr2_pos[1])
ax.plot_surface(X, Y, Z)

# Plot the source position
ax.scatter(source_position[0], source_position[1], source_position[2], color='green', label='Source')

# Show the plot
plt.show()
