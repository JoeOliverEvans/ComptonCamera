# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:42:42 2023

@author: dylan
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import sympy as smp
from scipy import constants

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
    fig, axes = plt.subplots(1,2)
    axes[0].plot(the, pdf_o)
    axes[0].set_title(r'$pdf(\theta, \phi)$', fontsize=20)
    axes[1].plot(the, cdf_o)
    axes[1].set_title(r'$cdf(\theta, \phi)$', fontsize=20)
    plt.show()

    return angle

def photon_path(start, angle, end_y):
    """
    Describes photon path based off of starting position and angle of direction
    """
    end = np.array([start[0] + np.cos(angle), end_y])
    return end

   
    # Starting position of photon
    start = [0, -2]
    
    # Initial direction of photon
    angle = np.pi * np.random.rand()
    
    # Scatter at first screen and get new end point
    end = photon_path(start, angle, 0)
    print(end)
    
    # Find end point of photon
    angle = kleinnishima(511000)
    end2 = photon_path(end, angle, 1)
    print(end2)
    
    # Plot the photon path
    plt.plot([start[0], end[0]], [start[1], end[1]])
    plt.plot([end[0], end2[0]], [end[1], end2[1]])
        
    # Plot the two screens
    plt.axhline(y=0, color='gray')
    plt.axhline(y=1, color='gray')
        
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def back_project(screen1, screen2, angle1, angle2):
    """
    Back-project to find the origin of the photon
    """
    x = (screen1[0] + screen2[0]) / 2
    y = (screen1[1] + screen2[1]) / 2
    theta = (angle2 - angle1) / 2 + angle1
    x0 = x - np.cos(theta)
    y0 = y - np.sin(theta)
    return [x0, y0]

#location = back_project(end, end2, angle, 0)
#print(location)
