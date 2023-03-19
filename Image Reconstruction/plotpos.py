#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:41:33 2023

@author: amberjones
"""

import matplotlib.pylab as plt
import numpy as np
import scipy.constants as constants
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in arcsin")

electron_mass = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 6)  # in MeV


def scatterAngle(scatter_energy_deposited, initial_energy = 0.662):
    """
    :param final_energy: keV
    :param initial_energy: keV
    :return: Compton Scattering Angle in radians
    """
    return np.arccos(1 - ((1/(initial_energy-scatter_energy_deposited)) - 1/initial_energy) * electron_mass)



data=pd.read_parquet(r'/Users/amberjones/Documents/IRdata/experimentalangle')

x_a = []
y_a = []
z_a = []

x_s = []
y_s = []
z_s = []

theta_c_all = []
print(data)

for i in range(len(data)):
    a_pos = data.iloc[i,1]
    s_pos = data.iloc[i,3]
    x_a.append(a_pos[0])
    y_a.append(a_pos[1])
    z_a.append(a_pos[2])

    x_s.append(s_pos[0])
    y_s.append(s_pos[1])
    z_s.append(s_pos[2])

    theta_c = scatterAngle(data.iloc[i,0])
    theta_c_all.append(theta_c)


n, bins, patches = plt.hist(x=theta_c_all, bins='auto', color='darkblue',
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Compton Angle (radians)')
plt.ylabel('Frequency')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.show()

