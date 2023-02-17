# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:13:02 2023

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy import constants
import random
import pandas as pd




# Define the parameters of the basic simulation
source_energy = 662000 # eV
n_photons = 5 # Number of photons to simulate on the scattering crystal 1


class KleinNishina:
    """Class that calculates a probability distribution from the Klein-Nishina
    equation. It can then turn it into polar and azimuthal scattering angles
    using this angle"""

    def __init__(self, E, no_of_points=10000):
        """Rearranges the Klein-Nishina distribution to form a cumulative probability
        distribution. Takes in the energy of the incident photon. Also takes in the
        no_of_points, which determines the number of possible angles that the 
        scattering angles can take. Increasing no_of_points increases execution
        time.
    
        Returns a cumulative density function and range of thetas, in order to 
        calculate scattering angles according to this distribution."""    
        self.E = E
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
        self.the = np.linspace(0, np.pi, no_of_points)
        self.pdf_o = pdf_omega_f(self.the, alp)    
        # Convert into cumulative density function
        self.cdf_o = np.cumsum(self.pdf_o)
        self.cdf_o /= self.cdf_o[-1]
    
    def calc_scattering_angles(self):
        """Takes in a photon energy (in eV), and calculates the scattering angle 
        caused by it Compton scattering off an electron. This is done by using the 
        Klein-Nishina equation in the form of a probability density function, and
        randomly assigning a scattering angle based on this distribution."""
        # Generate a random number that we can stick into the function to get a theta
        # value
        rand_o = np.random.rand()
        # Get the index of the nearest value to the random number
        i = np.argmin(np.abs(self.cdf_o - rand_o))
        # Extract the appropriate polar angle
        polar = self.the[int(i)]
        # Randomly assign an azimuthal angle. Since scattering is isotropic for
        # unpolarised photons, we can jut assign a random number.
        azimuthal = np.random.rand() * 2 * np.pi   
        return(polar, azimuthal)

# Create an instance of the Klein-Nishina class - get a probability distribution
# to get angles from.
kn = KleinNishina(source_energy)

#source position 
#placed randomly in a 15*15*15 box around the origin
source_pos = ([random.uniform(10, 20), random.uniform(10, 20), random.uniform(10, 20)])
print("The source position is:")
print(source_pos)

#scattering detector 1 position:
scatter_1_x = np.linspace(10, 20, 100)
scatter_1_y = np.linspace(10, 20, 100)
scatter_1_x, scatter_1_y = np.meshgrid(scatter_1_x, scatter_1_y)
scatter_1_z = 0*scatter_1_x + 30  

#absorbing detector 1 position:
absorber_1_x = np.linspace(10, 20, 100)
absorber_1_y = np.linspace(10, 20, 100)
absorber_1_x, absorber_1_y = np.meshgrid(absorber_1_x, absorber_1_y)
absorber_1_z = 0*absorber_1_x + 40 


#plots: 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Create arrays for the positions of hits on the 1st set of detectors
scat1_pos = np.zeros((n_photons, 3))
absorb1_pos = np.zeros((n_photons, 4))

for i in range(n_photons):
    
    #generating a random hit on scatterer 1 and plotting it
    scat1_pos[i, :] = 30, random.uniform(10, 20), random.uniform(10, 20)
    x_scat1 = [source_pos[0], scat1_pos[i][0]]
    y_scat1 = [source_pos[1], scat1_pos[i][1]]
    z_scat1 = [source_pos[2], scat1_pos[i][2]] 
    ax.plot(x_scat1, y_scat1, z_scat1)   

    angles = kn.calc_scattering_angles()
    theta1 = angles[0]
    azimuth1 = angles[1]
    absorb1_pos[i, :] = 40, scat1_pos[i][1] + 10*np.cos(theta1), scat1_pos[i][2] + 10*np.cos(azimuth1), theta1
    x_absorb1 = [scat1_pos[i][0], absorb1_pos[i][0]]
    y_absorb1 = [scat1_pos[i][1], absorb1_pos[i][1]]
    z_absorb1 = [scat1_pos[i][2], absorb1_pos[i][2]]
    ax.plot(x_absorb1, y_absorb1, z_absorb1)    



# Set of coordinates for the incident position on the scatterer 1
print("Set of coordinates for the incident position on scatterer 1:")
print(scat1_pos)
# Set of coordinates for the incident position on the absorber 1
print("Set of coordinates for the incident position on absorber 1:")
print(absorb1_pos)


#plotting source 
ax.scatter(source_pos[0], source_pos[1], source_pos[2], color='green', label='Source')
#plotting scattering detector 1
ax.plot_surface(scatter_1_z, scatter_1_y, scatter_1_x)
#plotting absorbing detector 1
ax.plot_surface(absorber_1_z, absorber_1_y, absorber_1_x)
plt.show()


# the file has columns: | scattering x pos |  scattering y pos | scattering z pos | absorbing x pos | absorbing y pos | absorbing z pos | scattering angle |
data = np.hstack((scat1_pos,absorb1_pos))
df = pd.DataFrame(data)
df.to_excel(excel_writer = r"copy filepath to excel file here .xls")
