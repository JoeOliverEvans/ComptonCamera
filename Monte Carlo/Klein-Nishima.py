import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy import constants

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
    
    def plot_dist(self):
        """Plot the probability and cumulative density functions for the given
        photon energy. Useful for visualising how the probability changes with
        polar angle."""
        
        fig, axes = plt.subplots(1,2)
        axes[0].plot(self.the, self.pdf_o)
        axes[0].set_title(r'$pdf(\theta, \phi)$', fontsize=20)
        axes[0].set_xlabel(r'$\theta$')
        axes[0].set_ylabel(r'Probability')
        
        axes[1].plot(self.the, self.cdf_o)
        axes[1].set_title(r'$cdf(\theta, \phi)$', fontsize=20)
        axes[1].set_xlabel(r'$\theta$')
        plt.show()

# Example execution
# kn = KleinNishina(661e3, 10000)

# for i in range(10):
#     polar, azimuthal = kn.calc_scattering_angles()
#     print(polar, azimuthal)

# kn.plot_dist()
