# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:45:56 2023

@author: Cara Clarke
"""

import numpy as np
import matplotlib.pyplot as plt


def cone(a1, a2, a3, v1, v2, v3, theta, X, Y):
    a = (v3**2 - (np.cos(theta))**2)
    b = 2*(v1*v3*X - v1*v3*a1 - (v3**2)*a3 + v2*v3*Y - v2*v3*a2 + a3*(np.cos(theta))**2) #2*(v1*v3*a1 + v2*v3*a2 - v1*v3*X - v2*v3*Y - a3*(np.cos(theta))**2)
    c = (v3**2)*(a3**2) + 2*v1*v3*a3*(a1-X) + 2*v2*v3*a3*(a2-Y) + (v1**2)*((X-a1)**2) + (v2**2)*((Y-a2)**2) + 2*v1*v2*(X-a1)*(Y-a2) - ((np.cos(theta))**2)*((a3**2) + (X-a1)**2 + (Y-a2)**2) #(X**2 - 2*a1*X + a1**2 + Y**2 - 2*a2*Y + a2**2 + a3**2) * (np.cos(theta))**2 - (v1**2)*(X-a1)**2 - (v2**2)*(Y-a2)**2 - (a3**2)*(v3**2) - 2*v1*v2*(X*Y - a2*a1 - X*a2 - Y*a1) + 2*v1*v3*a1*a3 + 2*v1*v3*X*a3 - 2*v2*v3*a2*a3 + 2*v2*v3*Y*a3
    Z = (-b + np.sqrt(abs((b ** 2) - (4 * (a) * c)))) / np.float(2 * (a))
    return Z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(-50.0, 50, 1)
y = np.arange(-50.0, 50, 1)
X, Y = np.meshgrid(x, y)
Z = cone(0, 0, 0, 0, 1, 1, np.pi/6, X, Y)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
