# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:57:05 2023

@author: CaraClarke
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.constants
import pandas as pd

#reading data
# df = pd.read_csv('data.csv') 
# r_absd = df[''] #absorption locations
# r_scatd = df[''] #scatter locations
# E_id = df[''] #initial energies
# E_fd = df[''] #final energies


h = scipy.constants.h
m_e = scipy.constants.m_e
c = scipy.constants.c

def scattering_angle(E_i, E_f):
    c_angle = (1 - ((m_e*(c**2))*(1/E_f-1/E_i)))
    return np.arccos(c_angle)

# def scattering_angle(a):
#     return a

def ap_vect(r_abs, r_scatter): #vector form cone apex
    # r_vect = r_abs - r_scatter 
    x_vect = r_abs[0] - r_scatter[0]
    y_vect = r_abs[1] - r_scatter[1]
    z_vect = r_abs[2] - r_scatter[2]
    n_x = x_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    n_y = y_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    n_z = z_vect/(np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    return (n_x, n_y, n_z)

def ax_ap_vect(r_abs, r_scatter): #angle between vector from cone apex and z
    x_vect = r_abs[0] - r_scatter[0]
    y_vect = r_abs[1] - r_scatter[1]
    z_vect = r_abs[2] - r_scatter[2]
    ang = np.arcos((z_vect)/(np.sqrt((x_vect**2 + y_vect**2 + z_vect**2))))
    return ang

def apex_point(r_scatter):
    return r_scatter

def x_vals(x_min, x_max, space): #defines x_vals so height and width of imaging space
    x_vals = np.arange(x_min, x_max, space)
    return x_vals

def conic(r_abs, r_scatter, ap_vect, scattering_angle, z_p, x_vals): #defines a conic section by putting z_p into equation of cone (could be completely wrong and plots two surfaces)
    a1 = apex_point(r_scatter)[0]
    a2 = apex_point(r_scatter)[1]
    a3 = apex_point(r_scatter)[2]
    v1, v2, v3 = ap_vect
    theta = scattering_angle
    x = x_vals
    a = (v2**2) - (np.cos(theta))**2
    #e1 = 2*(v1*v2*x - (v2**2)*a2 - v1*v2*a1 + v2*v3*z_p - v2*v3*a3)
    b = 2*(v1*v2*(x-a1) - (v2**2)*a2 + v2*v3*(z_p-a3) + a3*(np.cos(theta))**2)#e1 + 2*a2*(np.cos(theta))**2
    #e2 = ((np.cos(theta))**2)*((x**2) - 2*x*a1 + (a1**2) + (a2**2) + (a3**2) + (z_p**2) - 2*z_p*a3)
    #f2 = 2*(v1*v2*x*a2 - v1*v2*a1*a2 - v1*v3*(x-a1)*(z_p - a3) + v2*v3*a2*z_p - v2*v3*a2*a3)
    #g2 = (v1**2)*(x-a1)**2 + (v2**2)*a2 + (v3**2)*(z_p-a3)**2
    c = (v2**2)*(a2**2) + 2*v1*v2*a2*(a1-x) + 2*v2*v3*a2*(a3-z_p) + (v1**2)*((x-a1)**2) + (v3**2)*((z_p-a3)**2) + 2*v1*v3*(x-a1)*(z_p-a3) - ((np.cos(theta))**2)*((a2**2) + (x-a1)**2 + (z_p-a3)**2)#-(e2 + f2 - g2)
    # delta = b ** 2 - 4 *a* c
    # if delta.any() > 0 and delta.any() == 0:
    #     y1 = (-b + np.sqrt(delta)) / 2
    #     y2 = (-b - np.sqrt(delta)) / 2
    #     if y1.any() > 0 and y2.any() > 0:
    #         return y1, y2
    # return None
    y1 = (-b + np.sqrt(((b ** 2) - (4 * (a) * c)))) / (2 * (a))
    y2 = (-b - np.sqrt(((b ** 2) - (4 * (a) * c)))) / (2 * (a))
    # if b^2 - 4ac <0 we can't have it, if = 0 edge intersection?? if >0 this is what we're looking for
    return y1, y2

# def floor_div(var, seg):
#     var_idx = var//seg
#     return var_idx
    


z_p = 20 #estimated plane position, don't really know what I'm supposed to do with this
#scatter and absorbtion positions
r_abs1 = np.array([10, 10, 0])
r_scatter1 = np.array([5, 5, 10])
E_i = 662*(1.6*10**(-13))
E_f = 400*(1.6*10**(-13))
r_abs2 = np.array([10, 0, 0])
r_scatter2 = np.array([2, 6, 10])
E_i = 662*(1.6*10**(-13))
E_f2 = 430*(1.6*10**(-13))
x = x_vals(-100, 100, 0.001)

y1, y2 = conic(r_abs1, r_scatter1, ap_vect(r_abs1, r_scatter1), scattering_angle(E_i, E_f), z_p, x)
y3, y4 = conic(r_abs2, r_scatter2, ap_vect(r_abs2, r_scatter2), scattering_angle(E_i, E_f2), z_p, x)
# y3, y4 = ellipse(0, 0, 0, nx2, ny2, nz2, np.pi/20, z_p, x)
# y5, y6 = ellipse(0, 0, 0, nx3, ny3, nz3, np.pi/20, z_p, x)

#array of zeroes
zero = np.zeros(500)

# A = np.dstack([x, y1])
# print(A)

##indexing for intersections??
# seg = 0.1
# x_idx = floor_div(x_vals, seg)
# y1_idx = floor_div(y1, seg)
# y2_idx = floor_div(y2, seg)
# y3_idx = floor_div(y3, seg)
# y4_idx = floor_div(y4, seg)
# y5_idx = floor_div(y5, seg)
# y6_idx = floor_div(y6, seg)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1,  color='blue', alpha=0.9)
ax.plot(x, y2, color='red', alpha=0.9)
ax.plot(x, y3, color='blue', alpha=0.9)
ax.plot(x, y4, color='red', alpha=0.9)
# ax.plot(x3, y3)
# ax.grid(xgrid, ygrid)
# ax.plot(A)
# ax.plot(x_idx, y3_idx, color='blue', alpha=0.9)
# ax.plot(x_idx, y4_idx, color='red', alpha=0.9)
# ax.plot(x_idx, y5_idx, color='blue', alpha=0.9)
# ax.plot(x_idx, y6_idx, color='red', alpha=0.9)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_aspect('equal', adjustable='box')
plt.grid()

def calculate_conics(r_abs, r_scatter, E_f, E_i, z_p, x):
    a = []
    b = []
    for i in range(len(r_absd)): #for range of length the same as data given where r_absd = absorption posiiton data
        y1, y2 = conic(r_abs[i], r_scatter[i], ap_vect(r_abs[i], r_scatter[i]), scattering_angle(E_i[i], E_f[i]), z_p, x)
        a.append([y1])
        b.append([y2])
    return a, b

def plot_conics(calculate_conics, x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y1, y2 = calculate_conics
    ax.plot(x, y1,  color='blue', alpha=0.9)
    ax.plot(x, y2, color='red', alpha=0.9)
    plt.show()
        
    
##histogram plots for heat map intersections

# def hist(x, y, bins):
#     n, x_edges, y_edges = np.histogram2d(x, y, bins)
#     return n

# print(hist(np.isfinite(x), np.isfinite(y1), 20))

# n, xedges, yedges = np.histogram2d(np.isfinite(x), np.isfinite(y1), 20)
# n2, xedges2, yedges2 = np.histogram2d(np.isfinite(x), np.isfinite(y2), 20)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# # Plot histogram using pcolormesh
# # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
# ax1.pcolormesh(xedges, yedges, n, cmap='rainbow')
# ax1.plot(x, 2*np.log(x), 'k-')
# ax1.pcolormesh(xedges2, yedges2, n2, cmap='rainbow')
# ax1.plot(x, 2*np.log(x), 'k-')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_xlim(x.min(), x.max())
# ax1.set_ylim(y2.min(), y1.max())
# ax1.set_title('histogram2d')
# ax1.grid()