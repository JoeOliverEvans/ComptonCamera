# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:17:56 2023

@author: Cara
"""
import numpy as np
import matplotlib.pyplot as plt


#scatter and absorbtion positions
r_abs1 = np.array([40, 50, 0])
r_scatter1 = np.array([40, 50, 10])
r_abs2 = np.array([40, 50, 0])
r_scatter2 = np.array([40, 50, 10])
r_abs3 = np.array([40, 50, 0])
r_scatter3 = np.array([40, 50, 10])
# r_vect1 = r_abs1 - r_scatter1 
# x_vect1 = r_abs1[0] - r_scatter1[0]
# y_vect1 = r_abs1[1] - r_scatter1[1]
# z_vect1 = r_abs1[2] - r_scatter1[2]

#apex vectors
def ap_vect(r_abs, r_scatter):
    # r_vect = r_abs - r_scatter 
    x_vect = r_abs[0] - r_scatter[0]
    y_vect = r_abs[1] - r_scatter[1]
    z_vect = r_abs[2] - r_scatter[2]
    n_x = x_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    n_y = y_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    n_z = z_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    return n_x, n_y, n_z

nx1, ny1, nz1 = ap_vect(r_abs1, r_scatter1)
nx2, ny2, nz2 = ap_vect(r_abs2, r_scatter2)
nx3, ny3, nz3 = ap_vect(r_abs3, r_scatter3)

# n_x = x_vect1/(np.sqrt(x_vect1**2 + y_vect1**2 + z_vect1**2))
# n_y = y_vect1/(np.sqrt(x_vect1**2 + y_vect1**2 + z_vect1**2))
# n_z = z_vect1/(np.sqrt(x_vect1**2 + y_vect1**2 + z_vect1**2))

# #imaging plane
# for i in range (0,10):
#     z_p = i
#     if np.intersection(y1, y2, y3, y4) == nan:
#         i = i + 0.1
# z_p = 0

# y1, y2 = ellipse(0, 0, 0, nx1, ny1, nz1, np.pi/6, z_p)
# y3, y4 = ellipse(0, 0, 0, nx2, ny2, nz2, np.pi/6, z_p)

# while not y1.intersection(y3):
#     z_p +=1
    
    
# z_p = 10

# def plane(z_p):
#     return z_p

# #cone
# def cone(theta, r_vect, r_scatter):
#     r = (x, y, z)
#     r_2 = r_scatter - r
#     np.cos(theta) = np.dot(r_2, r_vect)/((np.linalg.norm(r_2))*(np.linalg.norm(r_vect)))
#     return r

# generate a grid of x and y values
# x = np.meshgrid(np.linspace(-10, 10, 500) #, np.linspace(-10, 10, 500))
x = np.linspace(-10, 10, 5000)

# def cone(a1, a2, a3, v1, v2, v3, theta):
#     e = (v3**2-(np.cos(theta))**2)
#     f = 2*(v1*v3*a1+v2*v3*a1-v1*v3*x-v2*v3*y-a3*(np.cos(theta))**2)
#     g = (x**2-2*a1*x+a1**2+y**2-2*a2*y+a2**2+a3**2)*(np.cos(theta))**2 - (v1**2)*(x-a1)**2 - (v2**2)*(y-a2)**2 - (a3**2)*(v3**2) - 2*v1*v2*(x*y - a2*a1 - x*a2 - y*a1) + 2*v1*v3*a1*a3 + 2*v1*v3*x*a3 - 2*v2*v3*a2*a3 + 2*v2*v3*y*a3
#     z1 = (-f + np.sqrt(f**2 + 4*e*g))/(2*e)
#     z2 = (-f - np.sqrt(f**2 + 4*e*g))/(2*e)
#     return z1, z2

def ellipse(a1, a2, a3, v1, v2, v3, theta, z_p):
    a = (v2**2) - (np.cos(theta))**2
    e1 = 2*(v1*v2*x - (v2**2)*a2 - v1*v2*a1 + v2*v3*z_p - v2*v3*a3)
    b = e1 + 2*a2*(np.cos(theta))**2
    e2 = ((np.cos(theta))**2)*((x**2) - 2*x*a1 + (a1**2) + (a2**2) + (a3**2) + (z_p**2) - 2*z_p*a3)
    f2 = 2*(v1*v2*x*a2 - v1*v2*a1*a2 - v1*v3*(x-a1)*(z_p - a3) + v2*v3*a2*z_p - v2*v3*a2*a3)
    g2 = (v1**2)*(x-a1)**2 + (v2**2)*a2 + (v3**2)*(z_p-a3)**2
    c = -(e2 + f2 - g2)
    y1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    y2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return y1, y2

# z1, z2 = cone(0, 0, 0, n_x, n_y, n_z, np.pi/6)

z_p = 3

y1, y2 = ellipse(0, 0, 0, nx1, ny1, nz1, np.pi/6, z_p)
y3, y4 = ellipse(0, 0, 0, nx2, ny2, nz2, np.pi/4, z_p)
y5, y6 = ellipse(1, -2, 0, nx3, ny3, nz3, np.pi/6, z_p)

x_idx = x//0.1
y1_idx = y1//0.1
y2_idx = y2//0.1
y3_idx = y3//0.1
y4_idx = y4//0.1
y5_idx = y5//0.1
y6_idx = y6//0.1

# while not y1.intersection(y3):
#     z_p +=1

# # generate a grid of x and y values
# x = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z1, color='blue', alpha=0.9)
# ax.plot_surface(x, y, z2, color='blue', alpha=0.9)
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_idx, y1_idx, color='blue', alpha=0.9)
ax.plot(x_idx, y2_idx, color='red', alpha=0.9)
ax.plot(x_idx, y3_idx, color='blue', alpha=0.9)
ax.plot(x_idx, y4_idx, color='red', alpha=0.9)
ax.plot(x_idx, y5_idx, color='blue', alpha=0.9)
ax.plot(x_idx, y6_idx, color='red', alpha=0.9)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal', adjustable='box')
plt.show()
 
print(np.intersect1d(y1_idx, y3_idx))