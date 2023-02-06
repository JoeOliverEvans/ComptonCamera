import numpy as np
import matplotlib.pyplot as plt
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
    # fig, axes = plt.subplots(1,2)
    # axes[0].plot(the, pdf_o)
    # axes[0].set_title(r'$pdf(\theta, \phi)$', fontsize=20)
    # axes[1].plot(the, cdf_o)
    # axes[1].set_title(r'$cdf(\theta, \phi)$', fontsize=20)
    # plt.show()

    return(angle)
