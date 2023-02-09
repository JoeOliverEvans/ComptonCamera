# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:53:20 2023

@author: dxe936
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the simulation
source_energy = 511 # keV
detector_thickness = 0.5 # cm
detector_efficiency = 0.5 # 50%
detector_resolution = 0.05 # keV
n_photons = 10 # Number of photons to simulate
scatter_screen_dimensions = [5, 5, 0.1] # 5x5 thin grid
screen_position = np.array([0, 0, 1]) # cm

#def klein_nishina(energy, theta):
    # Klein-Nishina formula
    #c = 1/137 # Fine structure constant
    #return (c/2) * (energy**2) * (1 + (1 - np.cos(theta))/2)

# Apply the Klein-Nishina formula. I found this on Wikipedia. The output is the changed energy of the photon
#def klein_nishina(energy, theta): 
    #m_e = 9.10938291e-31
    #c_light = 3e8
    #return energy*(1/(1+((energy/m_e*c_light**2)*(1-np.cos(theta)))))

#def inv_klein_nishina(energy, scattered_energies):
    #return np.arccos(1 + )



# Generate random angles for each photon
#angles = np.random.rand(n_photons, 3) # Creates vectors corresponding to each photons direction

# Calculate the scattered energy for each photon using the Klein-Nishina formula
#scattered_energies = [klein_nishina(source_energy, np.arccos(np.dot(angles[i], source_position))) for i in range(n_photons)]
# Dot product within np.arccos represents the angle of photons incident on the screen

# Simulate the scattering of each photon by the screen
#scattered_angles = []
#for i in range(n_photons):
   # if random.random() > scatter_screen_dimensions[0]/source_position[0]:
   #     scattered_angle = np.array([angles[i][0], angles[i][1] + random.uniform(-0.1, 0.1), angles[i][2] + random.uniform(-0.1, 0.1)])
   # else:
   #     scattered_angle = np.array([angles[i][0] + random.uniform(-0.1, 0.1), angles[i][1], angles[i][2] + random.uniform(-0.1, 0.1)])
   # scattered_angles.append(scattered_angle)
# This part is the next step ^^^ . Uses random dist. to change the direction of the photons 
# We want the photons to change direction following an inverse of the Klein-Nishina

# Simulate the detection of each photon by the Compton camera
#for i in range(n_photons): 
   # if random.random() < detector_efficiency and scattered_energies[i] > 0:
        #    detected_photons = [(scattered_angles[i], scattered_energies[i] + random.gauss(0, detector_resolution))]
# This applies the randomness of detection. Detected Photons array has angle and energy
#for i in range(len(detected_photons)):
   # angles_detected = [detected_photons[i][0] ]
   # energies_detected = [detected_photons[i][1]]
   
# Generate a random position for the source
source_position = np.random.rand(3) + 2*screen_position # + 2*screen_position ensures source is on other side

# Plot the source position, detector position, scattering material, and path of the photons
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 3D projection to show geometry of set-up I am a visual learner

# Plot the source position
ax.scatter(source_position[0], source_position[1], source_position[2], color='green', label='Source')

# Plot the detector position
detector_position = np.array([0, 0, 0]) # Detector is at (0,0,0) this is very helpful would recommend for everyone else to do this
ax.scatter(detector_position[0], detector_position[1], detector_position[2], color='blue', s=100)

# Plot the scattering material position
x = np.linspace(screen_position[0] - scatter_screen_dimensions[0]/2, screen_position[0] + scatter_screen_dimensions[0]/2, 10)
y = np.linspace(screen_position[1] - scatter_screen_dimensions[1]/2, screen_position[1] + scatter_screen_dimensions[1]/2, 10)
X, Y = np.meshgrid(x, y)
Z = np.full((10, 10), screen_position[2])
ax.plot_surface(X, Y, Z)

# Plot the path of the Photons
def scatter_photon(source_position, photon_direction):
    # Simulate the scattering of a photon in the material
    scattered_angle = np.random.uniform(0, 2 * np.pi) # This will be replaced with inv-klein_nishina
    azimuth_angle = np.random.uniform(0, 2 * np.pi)
    for i in range(n_photons):
        new_direction = np.array([np.sin(scattered_angle) * np.cos(azimuth_angle), #Spherical Polars
                             np.sin(scattered_angle) * np.sin(azimuth_angle),
                             np.cos(scattered_angle)])
    return source_position + scatter_screen_dimensions[2] * new_direction, new_direction

def calculate_solid_angle(detector_position, photon_position):
    # Calculate the solid angle swept out by the photon
    d = np.linalg.norm(photon_position - detector_position)
    return 2 * np.arccos(detector_thickness / d)

# Generate random directions for each photon
photon_directions = np.random.uniform(-1, 1, size=(n_photons, 3))
photon_directions /= np.linalg.norm(photon_directions, axis=1).reshape(-1, 1)

# Simulate the scattering of each photon
photon_positions = []
solid_angles = []
for i in range(n_photons):
    photon_pos, photon_dir = scatter_photon(source_position, photon_directions[i])
    if np.random.uniform() < detector_efficiency:
        photon_positions.append(photon_pos)
        solid_angles.append(calculate_solid_angle(detector_position, photon_pos))

photons_path = []
for i in range(n_photons):
    path = np.zeros((2,3))
    path[0,:] = source_position
    path[1,:] = screen_position
    photons_path.append(path)
    # This draws the line from source to screen
    
for i in range(n_photons):
        path[0,:] = screen_position
        path[1,:] = detector_position
        photons_path.append(path)
        # This draws line from screen to detector

for path in photons_path:
    ax.plot(path[:,0], path[:,1], path[:,2], c='red', linewidth=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()

# Use the solid angles to estimate the location of the source
source_estimate = np.sum(np.array(photon_positions) * np.array(solid_angles).reshape(-1, 1), axis=0) / np.sum(solid_angles)
# This line is complicated. By multiplying the photon positions on the detector by each respective
# solid angle and adding them together before dividing through by the sum of solid angles, it returns
# the point that the updated paths intersect at, giving a source estimate.
print("Estimated source position:", source_estimate)
print("Actual source position:", source_position)

