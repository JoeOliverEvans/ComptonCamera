# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:36:59 2023

@author: JackHocking
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def cone():
#     a = 0.5
#     b = -1
#     c = 0.5*(1 - X**2 - Y**2)
#     Z = (-b + np.sqrt(abs((b ** 2) - (4 * (a) * c))) / np.float(2 * (a)))
#     return Z                                     
         
def cone(a1, a2, a3, v1, v2, v3, theta, X, Y):
    a = (v3**2-(np.cos(theta))**2)
    b = 2*(v1*v3*a1+v2*v3*a1-v1*v3*X-v2*v3*Y-a3*(np.cos(theta))**2)
    c = (x**2-2*a1*X+a1**2+Y**2-2*a2*Y+a2**2+a3**2)*(np.cos(theta))**2 - (v1**2)*(x-a1)**2 - (v2**2)*(Y-a2)**2 - (a3**2)*(v3**2) - 2*v1*v2*(X*Y - a2*a1 - X*a2 - Y*a1) + 2*v1*v3*a1*a3 + 2*v1*v3*x*a3 - 2*v2*v3*a2*a3 + 2*v2*v3*y*a3
    Z = (-b + np.sqrt(abs((b ** 2) - (4 * (a) * c))) / np.float(2 * (a)))
    return Z                                                                     

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(-50.0, 50, 1)
y = np.arange(-50.0, 50, 1)
X, Y = np.meshgrid(x, y)
Z = cone(0, 0, 0, 0, 0, 1, np.pi/4, X, Y)


ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()