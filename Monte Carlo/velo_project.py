"""A simulation of particles passing through a VELO detector.

A number of particles are generated with uniformly sampled pseudorapidities, and sent through the
detector one by one, which returns their hit positions and number of hits. From these positions,
the transverse momenta of the particles are calculated and compared to the known true values to
determine the resolution of the detector.
The track reconstruction efficiency as a function of pseudorapidity is also calculated."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Particle:
    """Defines a particle. Takes an initial xyz coordinate, along with all three
    components of its momentum vector"""

    def __init__(self, p_x, p_y, p_z, x_0=0, y_0=0, z_0=0):
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z

        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0

class RightSensor:
    """Defines a right sensor in a VELO. Defines the boundaries for the sensor,
    to check whether a particle has passed through it or not"""

    def __init__(self):
        self.thickness = 0.2

    def check_if_inside(self, x_pos, y_pos):
        """Checks if a given x and y coordinate is within the VELO sensor. The
        values here were calculated from the figure given in velo.pdf, taking
        the centre of the VELO to be (0,0)"""

        # Checks if a particle is within given ranges. Units are mm.
        if (((-5.1 < x_pos <= 37.41) and (-33.51 < y_pos <= -5.1))
            or ((5.1 < x_pos <= 33.26) and (-5.1 < y_pos<= 33.51))):

            return True
        return False

class LeftSensor:
    """Defines a right sensor in a VELO. Defines the boundaries for the sensor,
    to check whether a particle has passed through it or not"""

    def __init__(self):
        self.thickness = 0.2

    def check_if_inside(self, x_pos, y_pos):
        """Checks if a given x and y coordinate is within the VELO sensor. The
        values here were calculated from the figure given in velo.pdf, taking
        the centre of the VELO to be (0,0)"""

        # Checks if a particle is within given ranges. Units are mm.
        if (((-33.26 < x_pos <= -5.1) and (-33.51 < y_pos <= 5.1))
                or ((-37.41 < x_pos < -5.1) and (5.1 < y_pos <= 33.51))):

            return True
        return False

class Velo:
    """Represents the VELO as a whole, including all left and right sensors.
    Stores the z positions of all sensors and contains a method that can
    calculate the number of hits that a particle passing through the detector
    would pick up"""

    def __init__(self):
        """Initialise an instance of the class. Defines the z positions of the left and right
        sensors"""

        self.pos_l_sensors = np.array([-277, -252, -227, -202, -132, -62, -37, -12, 13, 38, 63, 88,
                                       113, 138, 163, 188, 213, 238, 263, 325, 402, 497, 616, 661,
                                       706, 751])
        self.pos_r_sensors = np.array([-289, -264, -239, -214, -144, -74, -49, -24, 1, 26, 51, 76,
                                       101, 126, 151, 176, 201, 226, 251, 313, 390, 485, 604, 649,
                                       694, 739])

    def check_for_hit(self, sensor_z, sensor_obj, particle):
        """Checks if a given particle passes through a sensor, and thus outputs whether a 'hit'
        should be returned."""

        # Calculate the "time" at which the particle hits the middle of the sensor
        # - this is assumed to be the point at which the sensor makes a measurement
        time = (sensor_z - particle.z_0) / particle.p_z

        # Work out x and y values based on this "time"
        x_pos = particle.x_0 + particle.p_x * time
        y_pos = particle.y_0 + particle.p_y * time

        # Determine if there was a hit using the sensor instance.
        if sensor_obj.check_if_inside(x_pos, y_pos):
            return (x_pos, y_pos, sensor_z)
        return (0, 0, 0)

    def get_hits(self, particle, hit_eff, hit_res):
        """Calculates the number of hits the sensors will send, given a particle with specified
        momentum and initial position. Returns the number of hits."""

        # Check the sign of the particle's momentum, as that will affect which sensors it will pass
        # through
        if particle.p_z > 0:
            # Only get the sensors that the particle will pass through given its starting position
            l_sensors = self.pos_l_sensors[self.pos_l_sensors >= particle.z_0]
            r_sensors = self.pos_l_sensors[self.pos_r_sensors >= particle.z_0]

        elif particle.p_z < 0:
            l_sensors = self.pos_l_sensors[self.pos_l_sensors <= particle.z_0]
            r_sensors = self.pos_l_sensors[self.pos_r_sensors <= particle.z_0]

        # If the particle's z momentum is 0 then it will pass sideways out of the
        # detector and hit none, so no hits
        else:
            return 0, (0, 0, 0)

        # Initialise instances of the left and right sensors
        l_sensor = LeftSensor()
        r_sensor = RightSensor()

        # Create a 2D array with 3 columns - one for each coordinate
        hits_pos = np.zeros((len(l_sensors) + len(r_sensors), 3))
        # Loop through all left sensors and check if the x and y coords fall within it
        for i, sensor in enumerate(l_sensors):
            hits_pos[i] = self.check_for_hit(sensor, l_sensor, particle)
        # Do the same for the right sensors
        for i, sensor in enumerate(r_sensors):
            hits_pos[i + len(l_sensors)-1] = self.check_for_hit(sensor, r_sensor, particle)

        rows, cols = np.nonzero(hits_pos)
        hits_pos = hits_pos[np.unique(rows)]

        # Calculate an array of random numbers equal to the number of hits
        rand_vals = np.random.rand(len(hits_pos))
        # For each hit, check to see whether it was "detected" by seeing if its
        # random number is bigger than the hit efficiency for the VELO. Remove
        # the ones that didn't pass
        hits_pos = hits_pos[rand_vals < hit_eff]

        # The sensors have a maximum resolution determined by "hit_res". This
        # introduces a "smearing" of the data. This smearing effect is assumed
        # to have a Gaussian distribution, centered at the true hit position.
        # Note - only effects x and y coordinates, as z is known.
        hits_pos[:, 0:2] = np.random.normal(hits_pos[:, 0:2], scale=hit_res)

        # Return the no of hits and the position of the hits
        return len(hits_pos), hits_pos

def charged_particle_stream(velo, no_of_prtcls, eta_min, eta_max, p_mag):
    """Takes the number of particles, the minimum and maximum pseudorapidity
    values and the magnitude of the momentum to calculate particle momenta. Uses
    them to simulate particles with those momenta values passing through the
    VELO using the VELO class. Returns the measured transverse momenta for each
    particle, along with its the values for which particles were rejected and
    accepted"""

    # Uniformly sample values of phi and pseudo rapidity
    # Pseudo rap is the angle that the particle's momentum vector makes with the z-axis
    pseudo_raps = np.random.uniform(eta_min, eta_max, size=int(np.round(no_of_prtcls/100)))
    # Can determine the theoretical transverse momenta from pseudo-rapidity
    theory_p_ts = p_mag / np.cosh(pseudo_raps)

    # Phi is the angle that the particle's momentum vector makes with the y axis.
    phis = np.random.uniform(0, 360, size=100)

    # Initialise arrays here to speed up computation time.
    acc_and_rejs = np.zeros((len(pseudo_raps), 2))
    mean_msrd_p_ts = np.zeros((len(pseudo_raps), 2))

    # Loop over all pseudo rapidities
    for i, theory_p_t in enumerate(theory_p_ts):

        msrd_p_ts = np.zeros((len(phis), 2))
        for j, phi in enumerate(phis):
            # Generate a particle with a given pseudo rapidity and phi (which are converted into x,
            # y and z values)
            particle = Particle(theory_p_t * np.cos(phi),
                                theory_p_t * np.sin(phi),
                                theory_p_t * np.sinh(pseudo_raps[i]))

            # Get the hits from the velo class
            hits, hits_pos = velo.get_hits(particle, hit_eff=0.98, hit_res=0.0012)

            # A particle counts as being accepted if it has 3 or more hits.
            # Otherwise it is rejected
            if hits < 3:
                acc_and_rejs[i] += 0, 1

            else:
                # Only consider the particle if it has been accepted.
                acc_and_rejs[i] += 1, 0

                # Put the measured p_ts into an array
                msrd_p_ts[j] = calc_p_ts(hits, hits_pos, p_mag)

        # Remove zero values from the measured points. This gets rid of rejected hits.
        msrd_p_ts = msrd_p_ts[~np.all(msrd_p_ts == 0, axis=1)]

        # Take a mean of the p_ts for a given phi
        # Need to make sure that there have been some hits for that phi; if the
        # array is empty then np.mean throws an error
        if len(msrd_p_ts) != 0:
            mean_msrd_p_ts[i] = (np.mean(msrd_p_ts[:, 0], axis=0),
                                 np.sqrt(np.sum(np.square(msrd_p_ts[:, 1])))/len(msrd_p_ts))
        else:
            mean_msrd_p_ts[i] = 0, 0

    # Account for the cases where there are no hits for a given phi by removing them
    # from both measured and theoretical arrays. Also account for an error that occurs
    # where the covariance of a point is sometimes set to infinity.
    nohit_mask = ~np.all(mean_msrd_p_ts == 0, axis=1)
    infinity_mask = ~np.any(mean_msrd_p_ts == float('inf'), axis=1)
    # Take the AND of the two masks to get a combined mask
    mask = np.logical_and(nohit_mask, infinity_mask)

    mean_msrd_p_ts = mean_msrd_p_ts[mask]
    theory_p_ts = theory_p_ts[mask]

    return mean_msrd_p_ts, theory_p_ts, pseudo_raps, acc_and_rejs

def calc_p_ts(hits, hits_pos, p_mag):
    """Given a set of hit positions and a momentum magnitude, calculate the
    transverse momentum (and error) of the particle through performing hits of
    the x-z and y-z points and taking the gradients"""

    # Extract the measured p_x and p_y values
    p_xs = hits_pos[:, 0]
    p_ys = hits_pos[:, 1]
    p_zs = hits_pos[:, 2]

    # Fit the x and z components of the hits to a straight line to get the gradient
    # Sigma is set to 0.2 as the resolution of the measurements in x is known to be
    # 0.2mm
    est_error = np.full(hits, 0.2)
    # THESE LINES CAUSE A RUNTIME ERROR: "Covariance of the parameters could not be estimated"
    # Cannot work out how to fix it, so have added in a line later to correct for
    # the infinities that it creates
    xzfit, xzcov = curve_fit(linear_fit, p_zs, p_xs, p0=[0, p_xs[-1]/p_zs[-1]],
                             sigma=est_error, absolute_sigma=True)
    # Do the same for y and z
    yzfit, yzcov = curve_fit(linear_fit, p_zs, p_ys, p0=[0, p_ys[-1]/p_zs[-1]],
                             sigma=est_error, absolute_sigma=True)

    # Extract the gradients and respective errors from these fits
    m_x = xzfit[1]
    dm_x = np.sqrt(xzcov[1][1])
    m_y = yzfit[1]
    dm_y = np.sqrt(yzcov[1][1])

    # Find the transverse momentum and respective error
    p_t = p_mag * np.sqrt(1 - 1/(m_x**2 + m_y**2 + 1))

    dp_t = p_mag * np.sqrt(((m_x*dm_x)**2 + (m_y*dm_y)**2)
                           / ((m_x**2 + m_y**2)*(m_x**2 + m_y**2 + 1)**3))

    return p_t, dp_t

def plt_track_effs(pseudo_rap, acc_and_rejs, no_of_bins=30):
    """Groups given values of accepted and rejected values into bins, calculates the
    track recontruction efficiency for each bin and plots them against pseudorapidity"""

    # Put the accepted and rejected values into a specified number of bins
    track_effs = np.zeros(no_of_bins)

    # Get the width of the bars in the graph and initialise the array of x positions
    width = (max(pseudo_rap) - min(pseudo_rap)) / no_of_bins
    x_pos_raps = np.zeros(no_of_bins)
    min_val = min(pseudo_rap)

    # Loop through every bin
    for i in range(no_of_bins):
        # Create a mask so the right acc_rej values can be picked out for a certain
        # pseudorapidity bin
        mask = np.where((min_val <= pseudo_rap) & (pseudo_rap < min_val + width))

        sum_of_bin = np.sum(acc_and_rejs[mask], axis=0)
        # Calculate the track reconstruction efficiency for the given bin
        track_effs[i] = sum_of_bin[0] / (sum_of_bin[0] + sum_of_bin[1])

        # Calculate the x position of each bin
        x_pos_raps[i] = min_val
        min_val += width

    # Format and plot the graph
    fig, axes = plt.subplots(figsize=(5, 3.5), dpi=300)
    plt.bar(x_pos_raps, track_effs, width=width, linewidth=0.7, edgecolor="black",
            align="edge")

    plt.ylim(0, 1.01)
    plt.xlim(min(pseudo_rap) - 0.1, max(pseudo_rap) + 0.1)
    axes.set_xlabel(r"Pseudorapidity η")
    axes.set_ylabel("Track Reconstruction Efficiency")
    axes.set_title("Displaying how the Track Reconstruction Efficiency changes\n" +
              "with Pseudo Rapidity for the upgraded LHCb VELO")
    plt.show()

def plt_p_ts(theory_p_ts, msrd_p_ts, no_of_bins=25):
    """Plots the measured values of transverse momenta against those calculated
    from the "true" pseudorapidity values in grouped bins."""

    fig, axes = plt.subplots(figsize=(5, 3.5), dpi=300)

    width = (max(theory_p_ts) - min(theory_p_ts)) / no_of_bins
    min_val = min(theory_p_ts)
    # x and y positions that need to be plotted
    theory_pos = np.zeros(no_of_bins)
    msrd_pos = np.zeros(no_of_bins)

    # Break up the data into bins by applying a mask that takes slices of the arrays
    for i in range(no_of_bins):
        mask = np.where((min_val <= theory_p_ts) & (theory_p_ts < min_val + width))

        # Errors on msrd_p_ts have been left off, as they produce error bars too small
        # to see
        msrd_pos[i] = np.mean(msrd_p_ts[:, 0][mask])

        theory_pos[i] = min_val
        min_val += width

    # Plot the data points as groups
    plt.bar(theory_pos, msrd_pos, width=width, linewidth=0.7, edgecolor="black",
            label="Pt values", align="edge")

    # Fit the data points to a straight line to obtain a line of best fit
    # along with its gradient and y-intercept
    fit, cov = curve_fit(linear_fit, theory_p_ts, msrd_p_ts[:, 0],
                           sigma=msrd_p_ts[:, 1], absolute_sigma=True)

    yfit = fit[0] + fit[1]*theory_p_ts

    # Calculate the chi squared  and reduced chi squared values for the line of best fit
    chisqr = np.sum((msrd_p_ts[:, 0]
                     - linear_fit(theory_p_ts, inter=fit[0], slope=fit[1]))**2/msrd_p_ts[:, 1]**2)

    red_chisqr = chisqr / (len(msrd_p_ts[:, 0]) - 2) # Denominator is dof

    # Plot the line of best fit
    plt.plot(theory_p_ts, yfit, c="r",
             label="Line of best fit\n"
             + f"Gradient = {np.round(fit[1], 6)}±{np.round(np.sqrt(cov[1][1]), 6)}\n"
             + f"y-intercept = {np.round(fit[0], 6)}±{np.round(np.sqrt(cov[0][0]), 6)}GeV\n"
             + f"Reduced χ2 = {np.round(red_chisqr, 6)}")

    # Format the graph
    axes.set_xlabel("Theoretical Value of Pt (GeV)")
    axes.set_ylabel("Measured Value of Pt (GeV)")
    axes.set_title("Displaying how the values of Pt measured by the VELO\n differ"
                   + " from the theoretical values")
    plt.legend(fontsize=8)
    plt.show()

def linear_fit(x_pos, inter=0, slope=1):
    """Describes a straight line used by linear fits. Takes in x coordinate,
    and a gradient and y intercept. Returns y values."""

    y_pos = inter + x_pos*slope
    return y_pos

def main():
    """The main function. Runs the other functions in the script."""

    # Create an instance of the velo class
    velo = Velo()

    # Get the mean measured p_ts and the track efficiency for the given values
    msrd_p_ts, theory_p_ts, pseudo_raps, acc_and_rejs = charged_particle_stream(
        velo, no_of_prtcls=100000, eta_min=0, eta_max=7, p_mag=10)

    # Plot the track reconstruction efficiency against pseudorapidity
    plt_track_effs(pseudo_raps, acc_and_rejs)
    # Plot the measured transverse momenta against those calculated theoretically
    plt_p_ts(theory_p_ts, msrd_p_ts)

    # Calculate the root mean squared value of the difference between the theoretical and measured
    # values. This can provide a value for the resolution of the VELO.
    squared_diff = np.square(theory_p_ts - msrd_p_ts[:, 0])
    rms = np.sqrt(np.mean(squared_diff))

    # Propagate error from the measured p_ts to get the error on the resolution
    rms_err = np.sqrt(np.sum(squared_diff*np.square(msrd_p_ts[:, 1]))
                      / (np.sum(squared_diff)*len(squared_diff)))

    print(f"The resolution is {rms}±{rms_err}GeV")

if __name__ == "__main__":
    main()
