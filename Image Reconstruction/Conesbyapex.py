# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:48:55 2023

@author: CaraClarke
"""
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cone(a1, a2, a3, v1, v2, v3, theta):
    e = (v3**2-(np.cos(theta))**2)
    f = 2*(v1*v3*a1+v2*v3*a1-v1*v3*x-v2*v3*y-a3*(np.cos(theta))**2)
    g = (x**2-2*a1*x+a1**2+y**2-2*a2*y+a2**2+a3**2)*(np.cos(theta))**2 - (v1**2)*(x-a1)**2 - (v2**2)*(y-a2)**2 - (a3**2)*(v3**2) - 2*v1*v2*(x*y - a2*a1 - x*a2 - y*a1) + 2*v1*v3*a1*a3 + 2*v1*v3*x*a3 - 2*v2*v3*a2*a3 + 2*v2*v3*y*a3
    z1 = (-f + np.sqrt(f**2 + 4*e*g))/(2*e)
    z2 = (-f - np.sqrt(f**2 + 4*e*g))/(2*e)
    return z1, z2

# generate a grid of x and y values
x, y = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

print(x, y)

# calculate the values of z1 and z2 using the cone equation
z1, z2 = cone(0, 0, 0, 1, 0, 0, np.pi/6)

# plot the surface of the cone
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z1, color='blue', alpha=0.9)
ax.plot_surface(x, y, z2, color='blue', alpha=0.9)
plt.show()