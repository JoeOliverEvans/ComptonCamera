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
import scipy.stats as stats

h = scipy.constants.h
m_e = scipy.constants.m_e
cs = scipy.constants.c
e = scipy.constants.e

#reading data
df = pd.read_parquet('dictionarytest.parquet') 
# df = pd.read_excel('data.xlsx') 
r_absd = df['absorption locations'] #absorption locations
r_scatd = df['scatter locations'] #scatter locations
E_is = 0.662*(10**6)*scipy.constants.e #initial energy 662kev
E_fd = df['absorption energies']*(10**6)*scipy.constants.e #final energies

#filter data for just scatter-absorptions of just scattering to absorbing detectors
#get data for when absorption position is one of absorbers and scattering position is one of scatterers
# z_scatd = z position of absorption detectors
# z_absd = z position of scattering detectors
data = []
for i in range(len(df)):
    if r_scatd[i][2] < 50 and r_absd[i][2] > 50:
        data.append(df.iloc[i])
    # elif r_scatd[i][2] > 500:
# r_absd2 = data['absorption locations']
# data2 = []
# for i in range(len(data)):        
#     if r_absd2[i][2] > 500:
#         data2.append(df.iloc[i])

s_data = []
for i in range(len(data)):
    s = data[i]['scatter locations']
    s_data.append(s)
a_data = []
for i in range(len(data)):
    a = data[i]['absorption locations']
    a_data.append(a)
Ef_data = []
for i in range(len(data)):
        E = (data[i]['absorption energies'])*(10**6)*scipy.constants.e
        Ef_data.append(E)


def scattering_angle(E_i, E_f):
    c_angle = (1 - ((m_e*(cs**2))*(1/E_f-1/E_i)))
    return np.arccos(c_angle)

# def scattering_angle(a):
#     return a

def ap_vect(r_abs, r_scatter): #vector form cone apex
    # r_vect = r_abs - r_scatter 
    x_vect = int(r_abs[0]) - int(r_scatter[0])
    y_vect = int(r_abs[1]) - int(r_scatter[1])
    z_vect = int(r_abs[2]) - int(r_scatter[2])
    n = (np.sqrt(x_vect**2 + y_vect**2 + z_vect**2))
    n_x = x_vect/n
    n_y = y_vect/n
    n_z = z_vect/n
    return (n_x, n_y, n_z)

def ax_ap_vect(r_abs, r_scatter): #angle between vector from cone apex and z
    x_vect = r_abs[0] - r_scatter[0]
    y_vect = r_abs[1] - r_scatter[1]
    z_vect = r_abs[2] - r_scatter[2]
    ang = np.arcos((z_vect)/(np.sqrt((x_vect**2 + y_vect**2 + z_vect**2))))
    return ang

def apex_point(r_scatter):
    return r_scatter[0], r_scatter[1], r_scatter[2]

def x_vals(x_min, x_max, space): #defines x_vals so height and width of imaging space
    x_vals = np.arange(x_min, x_max, space)
    return x_vals

def conic(r_abs, r_scatter, ap_vect, scattering_angle, z_p, x_vals): #defines a conic section by putting z_p into equation of cone (could be completely wrong and plots two surfaces)
    a1, a2, a3 = apex_point(r_scatter)
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
    y1_ = y1[~np.isnan(y1)]
    y2 = (-b - np.sqrt(((b ** 2) - (4 * (a) * c)))) / (2 * (a))
    y2_ = y2[~np.isnan(y2)]
    x_vals_ = x[~np.isnan(y1)]
    # x_l = np.linspace(0, x_vals_, len(x_vals_))
    # y1__ = np.interp(x_l, x_vals_, y1_)
    # y2__ = np.interp(x_l, x_vals_, y2_)
    # if b^2 - 4ac <0 we can't have it, if = 0 edge intersection?? if >0 this is what we're looking for
    
    return y1_, y2_, x_vals_

# def floor_div(var, seg):
#     var_idx = var//seg
#     return var_idx
    


z_p = 15.5 #where the z-plane of the source is supposed to be #estimated plane position, don't really know what I'm supposed to do with this
# # #hard coded scatter and absorbtion positions (3 events)
# r_abs1 = np.array([-10.8615, 54.7378, 752.1139999999999])
# r_scatter1 = np.array([21.8272, 36.5786, 415.97900000000004])
# E_i1 = 0.662*(1.6*10**(-13))
# E_f1 = 0.407548*(1.6*10**(-13))
# # r_abs1 = [r_absd[0][0], r_absd[0][1], r_absd[0][2]]
# # r_scatter1 = [r_scatd[0][0], r_scatd[0][1], r_scatd[0][2]]
# # E_i1 = E_is
# # # E_f1 = E_fd[0]
# r_abs2 = np.array([-10.8615, 54.7378, 752.1139999999999])
# r_scatter2 = np.array([23.0623, 35.9737, 417.564])
# E_i2 = 0.662*(1.6*10**(-13))
# E_f2 = 0.407548*(1.6*10**(-13))
# r_abs3 = np.array([22.2761, 37.7457, 396.92])
# r_scatter3 = np.array([-20.2686, 36.7433, 400.541])
# E_i3 = 0.662*(1.6*10**(-13))
# E_f3 = 0.268266*(1.6*10**(-13))
# r_abs4 = np.array([-18.8499, 28.7753, 415.602])
# r_scatter4 = np.array([26.116999999999997, 22.7321, 413.923])
# E_i4 = 0.662*(1.6*10**(-13))
# E_f4 = 0.276506*(1.6*10**(-13))
# # r_abs5 = np.array([-10, 0, 80])
# r_scatter5 = np.array([10, 0, 80])
# E_i5 = 662*(1.6*10**(-13))
# E_f5 = 289.905*(1.6*10**(-13))
# r_abs6 = np.array([-10, 0, 80])
# r_scatter6 = np.array([0, -10, 80])
# E_i6 = 662*(1.6*10**(-13))
# E_f6 = 280.492*(1.6*10**(-13))
# r_abs7 = np.array([10, 0, 80])
# r_scatter7 = np.array([0, 10, 80])
# E_i7 = 662*(1.6*10**(-13))
# E_f7 = 250.247*(1.6*10**(-13))


x = x_vals(-10, 10, 0.01)


# #hard coded y-values
# y1, y2, x2 = conic(r_abs1, r_scatter1, ap_vect(r_abs1, r_scatter1), scattering_angle(E_i1, E_f1), z_p, x)
# y3, y4, x3 = conic(r_abs2, r_scatter2, ap_vect(r_abs2, r_scatter2), scattering_angle(E_i2, E_f2), z_p, x)
# y5, y6, x4 = conic(r_abs3, r_scatter3, ap_vect(r_abs3, r_scatter3), scattering_angle(E_i3, E_f3), z_p, x)
# y7, y8, x5 = conic(r_abs4, r_scatter4, ap_vect(r_abs4, r_scatter4), scattering_angle(E_i4, E_f4), z_p, x)
# y9, y10, x6 = conic(r_abs5, r_scatter5, ap_vect(r_abs5, r_scatter5), scattering_angle(E_i5, E_f5), z_p, x)
# y11, y12, x7 = conic(r_abs6, r_scatter6, ap_vect(r_abs6, r_scatter6), scattering_angle(E_i6, E_f6), z_p, x)
# y13, y14, x8 = conic(r_abs7, r_scatter7, ap_vect(r_abs7, r_scatter7), scattering_angle(E_i7, E_f7), z_p, x)

## plotting hard coded data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x2, y1,  color='blue', alpha=0.9)
# ax.plot(x2, y2, color='blue', alpha=0.9)
# ax.plot(x3, y3, color='red', alpha=0.9)
# ax.plot(x3, y4, color='red', alpha=0.9)
# ax.plot(x4, y5, color='green', alpha=0.9)
# ax.plot(x4, y6, color='green', alpha=0.9)
# ax.plot(x5, y7,  color='blue', alpha=0.9)
# ax.plot(x5, y8, color='blue', alpha=0.9)
# ax.plot(x6, y9, color='red', alpha=0.9)
# ax.plot(x6, y10, color='red', alpha=0.9)
# ax.plot(x7, y11, color='green', alpha=0.9)
# ax.plot(x7, y12, color='green', alpha=0.9)
# ax.plot(x8, y13,  color='blue', alpha=0.9)
# ax.plot(x8, y14, color='blue', alpha=0.9)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_aspect('equal', adjustable='box')
# plt.grid()

def calculate_conics(r_abs, r_scatter, E_f, E_i, z_p, x):
    a = []
    b = []
    c = []
    for i in range(10):#(len(r_absd)): #for range of length the same as data given where r_absd = absorption posiiton data
        y1, y2, x_s = conic(r_abs[i], r_scatter[i], ap_vect(r_abs[i], r_scatter[i]), scattering_angle(E_i, E_f[i]), z_p, x)
        a.append([y1])
        b.append([y2])
        c.append([x_s])
    return a, b, c
# a = []
# b = []
# c = []
# for i in range(2, 4):#(len(r_absd)):
#     #for range of length the same as data given where r_absd = absorption posiiton data
#     y1, y2, x_s = conic(r_absd[i], r_scatd[i], ap_vect(r_absd[i], r_scatd[i]), scattering_angle(E_is, E_fd[i]), z_p, x)
#     a.append([y1])
#     b.append([y2])
#     c.append([x_s])

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(c, a,  color='blue', alpha=0.9)
# ax.plot(c, b, color='blue', alpha=0.9)
# plt.show()
a, b, c = calculate_conics(a_data, s_data, Ef_data, E_is, z_p, x)

def plot_conics(a, b, c):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # a, b, c = calculate_conics
    for i in range(len(a)):
        ax.plot(c[i][0], a[i][0],  color='blue', alpha=0.9)
        ax.plot(c[i][0], b[i][0], color='blue', alpha=0.9)
    # ax.plot(c, a,  color='blue', alpha=0.9)
    # ax.plot(c, b, color='blue', alpha=0.9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatterer-to-Absorber Conic Projections')
    fig.savefig('conicsproj.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()

plot_conics(a, b, c)
flat_a = []
for i in range(len(a)):
    for j in range(len(a[i][0])):
        flat_a.append(a[i][0][j])
flat_b = []
for i in range(len(b)):
    for j in range(len(b[i][0])):
        flat_b.append(b[i][0][j])
flat_c = []
for i in range(len(c)):
    for j in range(len(c[i][0])):
        flat_c.append(c[i][0][j])
     

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(flat_c, flat_a,  color='blue', alpha=0.9)
# ax.plot(c[9][0], b[9][0],  color='blue', alpha=0.9)
# plot_conics(calculate_conics(r_absd, r_scatd, E_id, E_fd, z_p, x))

# def plot_density(d1, d2): #very very slow
#     xy = np.vstack([d1, d2])
#     kde = stats.gaussian_kde(xy)
#     density = kde(xy)
#     fig7 = plt.figure()
#     ax7 = fig7.add_subplot(111)
#     scatter = ax7.scatter(d1, d2, c = density)
#     fig7.colorbar(scatter, location='left')
#     max_pos = xy.T[np.argmax(density)]
#     ax7.set_xlabel('X')
#     ax7.set_ylabel('Y')
#     ax7.set_zlabel('Z')
#     fig7.savefig('absorber.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#     plt.show()
#     return max_pos

# plot_density(flat_c, flat_a)
##histogram plots for heat map intersections

#waste of time and doesn't work

def hist(x, y):
    xedges = np.linspace(min(x), max(x))
    yedges = np.linspace(min(y), max(y))
    n, x_edges, y_edges = np.histogram2d(x, y, bins=(xedges, yedges))
    return n

h1 = hist(flat_c, flat_a)
h2 = hist(flat_c, flat_b)
h = h1 + h2
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.imshow(h.T)

def plot_density(d1, d2):
    xy = np.vstack([d1, d2])
    kde = stats.gaussian_kde(xy)
    density = kde(xy)
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)
    scatter = ax7.scatter(d1, d2, c = density)
    fig7.colorbar(scatter, location='left')
    max_pos = xy.T[np.argmax(density)]
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_title('Density Plot of Conic Sections for Scatterer-to-Absorber Events')
    ax7.set_ylim(0, 15)
    fig7.savefig('conics.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()
    return max_pos

print(plot_density(flat_c, flat_a))

# def plot_hist2(x, y):
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot(111)

# def plot_hist(a, b, c):
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot(111)
#     # a, b, c = calculate_conics
#     histogram = []
#     for i in range(7):
#         h1 = hist(c[i][0], a[i][0])
#         h2 = hist(c[i][0], b[i][0])
#         h = h1 + h2
#         histogram.append(h)
#     histogram2 = 0
#     for i in range(len(histogram)):
#         histogram2 += histogram[i]
#     ax2.imshow(histogram2.T)
#     plt.show()
#     return histogram2

# plot_hist(a, b, c) #nope waste of time and doesn't work
    
# histogram = []
# for i in range(2):
#     h1 = hist(c[i][0], a[i][0])
#     h2 = hist(c[i][0], b[i][0])
#     h = h1 + h2
#     histogram.append(h)
# histogram = []
# histogram2 = 0
# h1 = hist(c[0][0], a[0][0])
# h2 = hist(c[0][0], b[0][0])
# h3 = hist(c[1][0], a[1][0])
# h4 = hist(c[1][0], b[1][0])
# h = h1 + h2 + h3 + h4
# histogram.append([h1, h2, h3, h4])
# for i in range(len(histogram[0])):
#     histogram2 += histogram[i]
# print(histogram2)
# # fig2 = plt.figure()
# # ax1 = fig2.add_subplot(111)
# # ax1.imshow(h.T)

