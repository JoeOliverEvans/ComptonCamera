# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:19:08 2023

@author: dylan
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import sympy as smp
from scipy import constants


# Define the parameters of the simulation
source_energy = 511 # keV
detector_thickness = 0.5 # cm
detector_efficiency = 0.5 # 50%
detector_resolution = 0.05 # keV
n_photons = 10
 # Number of photons to simulate

# Define the screen and detector positions
screen_pos = np.array([0, 50, 0])
detector_pos = np.array([0, 100, 0])
scatter_screen_dimensions = [5, 0.5, 5]
detector_dimensions = [5, 0.5, 5]
scr2_pos = np.array([50, 0, 0])
scr2_dim = [0.5, 5, 5]
det2_pos = np.array([100, 0, 0])
det2_dim = [0.5, 5, 5]

def kleinnishima(E):
    """Takes in a photon energy (in eV), and calculates the scattering angle 
    caused by it Compton scattering off an electron. This is done by using the 
    Klein-Nishima equation in the form of a probability density function, and
    randomly assigning a scattering angle based on this distribution."""
    
    # Get alpha by dividing by the mass of the electron times c squared
    alp = (E * constants.e)/(constants.electron_mass * constants.c**2)
    
    # Radius of electron
    r_e = 2.8179403262e-15
    
    # Currently this works by actually performing the integral shown in
    # https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid30.ipynb
    # every single time. This is very inefficient but I haven't found a way to
    # the exact same results using another method yet.
    theta, alpha = smp.symbols(r'\theta \alpha', real=True, positive=True)
    dsdo = smp.Rational(r_e,2)*(1+smp.cos(theta)**2)/(1+alpha*(1-smp.cos(theta)))**2 * \
                ( 1 + alpha**2 * (1-smp.cos(theta))**2 / ((1+smp.cos(theta)**2)*(1+alpha*(1-smp.cos(theta)))))
    dsdt = 2*smp.pi*dsdo*smp.sin(theta)
    dsdt = dsdt.simplify()  
  
    s = smp.integrate(dsdt, (theta, 0, smp.pi))
    s = s.simplify().simplify()

    pdf_omega = dsdo / s
    pdf_omega=pdf_omega.simplify()

    pdf_omega_f = smp.lambdify([theta,alpha], pdf_omega)
    
    # Generate many values for theta and put them into the pdf
    the = np.linspace(0, np.pi, 10000)
    pdf_o = pdf_omega_f(the, alp)

    # Convert into cumulative density function
    cdf_o = np.cumsum(pdf_o)
    
    # Generate a random number that we can stick into the function to get a theta
    # value
    rand_o = np.random.rand()

    # Get the index of the nearest value to the random number
    i = np.argmin(np.abs(cdf_o - rand_o*np.sum(pdf_o)))

    # Extract the appropriate angle
    angle = the[int(i)]
    
    # # Here's a plot to help show what the pdf and cdf look like
    #fig, axes = plt.subplots(1,2)
    #axes[0].plot(the, pdf_o)
    #axes[0].set_title(r'$pdf(\theta, \phi)$', fontsize=20)
    #axes[1].plot(the, cdf_o)
    #axes[1].set_title(r'$cdf(\theta, \phi)$', fontsize=20)
    #plt.show()

    return angle


def photon_path(start, angle, end_y):
    """
    Describes photon path based off of starting position and angle of direction
    """
    end = np.array([start[0] + np.cos(angle[0]), end_y, start[2] + np.cos(angle[1])])
    return end

# Starting position of photon
start = [0, 0, 0]
    
# Initial direction of photons
num_photons = 10 # Number of photons to simulate
angles = np.pi * np.random.rand(num_photons, 2)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
for angle in angles:
        # Photons travel to first screen
        end = photon_path(start*screen_pos[1], angle, screen_pos[1])
        
        # Photons scatter obeying KN then travel to detectors y position
        new_angle = np.array([kleinnishima(511000), kleinnishima(511000)])
        end_y = detector_pos[1]
        end2 = photon_path(end, new_angle, end_y)
        
        # Plot the photon path
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
        ax.plot([end[0], end2[0]], [end[1], end2[1]], [end[2], end2[2]])
        
def photon_path2(start, angle, end_x):
    """
    Describes photon path towards second screen/ detector based off of starting position and angle of direction
    """
    end = np.array([end_x, start[1] + np.cos(angle[0]), start[2] + np.cos(angle[1])])
    return end

for angle in angles:
        # Photons travel to first screen
        end = photon_path(start*screen_pos[0], angle, screen_pos[0])
        
        # Photons scatter obeying KN then travel to detectors y position
        new_angle = np.array([kleinnishima(511000), kleinnishima(511000)])
        end_x = det2_pos[0]
        end2 = photon_path(end, new_angle, end_x)
        
        # Plot the photon path
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
        ax.plot([end[0], end2[0]], [end[1], end2[1]], [end[2], end2[2]])
        

# Plot the detector position
detector_position = detector_pos
x = np.linspace(detector_position[0] - detector_dimensions[0]/2, detector_position[0] + detector_dimensions[0]/2, 10)
z = np.linspace(detector_position[2] - detector_dimensions[2]/2, detector_position[2] + detector_dimensions[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), detector_position[1])
ax.plot_surface(X, Y, Z)

"""
# Plot the scattering material position
x = np.linspace(screen_pos[0] - scatter_screen_dimensions[0]/2, screen_pos[0] + scatter_screen_dimensions[0]/2, 10)
z = np.linspace(screen_pos[2] - scatter_screen_dimensions[2]/2, screen_pos[2] + scatter_screen_dimensions[2]/2, 10)
X, Z = np.meshgrid(x, z)
Y = np.full((10, 10), screen_pos[1])
ax.plot_surface(X, Y, Z)
"""
"""
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
"""

