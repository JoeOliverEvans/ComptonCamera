# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:15:30 2023

@author: CaraCLarke
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

#get data
df = pd.read_csv('newdata.csv', sep=';')
df.to_parquet('newdata.parquet')
df2 = pd.read_parquet('newdata.parquet') 
det = df2['CHANNEL']
X = df2['X']
Y = df2['Y']
Z = df2['Z']
scatter_pos = df2.query('`INTERACTION TYPE` == "compt"')
abs_pos = df2.query('`INTERACTION TYPE` == "phot"')

#scattering positions
x_s = scatter_pos['X']
y_s = scatter_pos['Y']
z_s = scatter_pos['Z']
#absorption position
x_a = abs_pos['X']
y_a = abs_pos['Y']
z_a = abs_pos['Z']

#Plot the event posiitons
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_s, y_s, z_s)
ax.scatter(x_a, y_a, z_a) #c=c, cmap=plt.cm.twilight_shifted)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#detectors
d_1 = df2.query('CHANNEL == 0')
d_2 = df2.query('CHANNEL == 1')
d_3 = df2.query('CHANNEL == 2')
d_4 = df2.query('CHANNEL == 3')
d_5 = df2.query('CHANNEL == 4')
d_6 = df2.query('CHANNEL == 5')
d_7 = df2.query('CHANNEL == 6')
d_8 = df2.query('CHANNEL == 7')



#detector absorption
d_1_a = abs_pos.query('CHANNEL == 0')
x_a_1 = d_1_a['X']
y_a_1 = d_1_a['Y']
z_a_1 = d_1_a['Z']
d_2_a = abs_pos.query('CHANNEL == 1')
x_a_2 = d_2_a['X']
y_a_2 = d_2_a['Y']
z_a_2 = d_2_a['Z']
d_3_a = abs_pos.query('CHANNEL == 2')
x_a_3 = d_3_a['X']
y_a_3 = d_3_a['Y']
z_a_3 = d_3_a['Z']
d_4_a = abs_pos.query('CHANNEL == 3')
x_a_4 = d_4_a['X']
y_a_4 = d_4_a['Y']
z_a_4 = d_4_a['Z']
d_5_a = abs_pos.query('CHANNEL == 4')
x_a_5 = d_5_a['X']
y_a_5 = d_5_a['Y']
z_a_5 = d_5_a['Z']
d_6_a = abs_pos.query('CHANNEL == 5')
x_a_6 = d_6_a['X']
y_a_6 = d_6_a['Y']
z_a_6 = d_6_a['Z']
d_7_a = abs_pos.query('CHANNEL == 6')
x_a_7 = d_7_a['X']
y_a_7 = d_7_a['Y']
z_a_7 = d_7_a['Z']
d_8_a = abs_pos.query('CHANNEL == 7')
x_a_8 = d_8_a['X']
y_a_8 = d_8_a['Y']
z_a_8 = d_8_a['Z']

#detector scatter
d_1_s = scatter_pos.query('CHANNEL == 0')
x_s_1 = d_1_s['X']
y_s_1 = d_1_s['Y']
z_s_1 = d_1_s['Z']
d_2_s = scatter_pos.query('CHANNEL == 1')
x_s_2 = d_2_s['X']
y_s_2 = d_2_s['Y']
z_s_2 = d_2_s['Z']
d_3_s = scatter_pos.query('CHANNEL == 2')
x_s_3 = d_3_s['X']
y_s_3 = d_3_s['Y']
z_s_3 = d_3_s['Z']
d_4_s = scatter_pos.query('CHANNEL == 3')
x_s_4 = d_4_s['X']
y_s_4 = d_4_s['Y']
z_s_4 = d_4_s['Z']
d_5_s = scatter_pos.query('CHANNEL == 4')
x_s_5 = d_5_s['X']
y_s_5 = d_5_s['Y']
z_s_5 = d_5_s['Z']
d_6_s = scatter_pos.query('CHANNEL == 5')
x_s_6 = d_6_s['X']
y_s_6 = d_6_s['Y']
z_s_6 = d_6_s['Z']
d_7_s = scatter_pos.query('CHANNEL == 6')
x_s_7 = d_7_s['X']
y_s_7 = d_7_s['Y']
z_s_7 = d_7_s['Z']
d_8_s = scatter_pos.query('CHANNEL == 7')
x_s_8 = d_8_s['X']
y_s_8 = d_8_s['Y']
z_s_8 = d_8_s['Z']

#plot of the frequecies of detections of photons at each detector 
freqs = np.array([len(d_1), len(d_2), len(d_3), len(d_4), len(d_5), len(d_6), len(d_7), len(d_8)])
det_no = np.arange(0, 8, 1)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(det_no, freqs)
# ax2.bar(det_no, freqs)
ax2.set_xlabel('detector number')
ax2.set_ylabel('frequency')

def plot_detector(d): #plot of single detector
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(d['X'], d['Y'], d['Z'])

plot_detector(d_1) #example with detector channel 0

data = np.dstack((X, Y, Z))


#looking at a 3D histogram but not plotted - not sure how to do this
def hist3d(x, y, z, bins_):
    hist, edges = np.histogramdd((x, y, z), bins_)
    return hist, edges

nbins = 10
hist_, edges = hist3d(X, Y, Z, nbins)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
# ax4.scatter(X, Y, Z, 'k.', alpha=0.3)

#Use one less than bin edges to give rough bin location
X_, Y_ = np.meshgrid(edges[0][:-1],edges[1][:-1])

#Loop over range of slice locations (default histogram uses 10 bins)
for ct in [0, 2, 5, 7, 9]: 
    cs = ax4.contourf(X_,Y_,hist_[:,:,ct], 
                      zdir='z', 
                      offset=edges[2][ct], 
                      level=100, 
                      cmap=plt.cm.RdYlBu_r, 
                      alpha=0.5)

# # Set the axis labels
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')

# # Show the plot
# plt.show()


fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.scatter(X, Y)
# ax5.scatter(Y, Z)
# ax5.scatter(X, Z)

def hist(x, y, bins): #2d histogram of X and Y positions
    # xedges = np.linspace(min(x), max(x))
    # yedges = np.linspace(min(y), max(y))
    n, x_edges, y_edges = np.histogram2d(x, y, bins)#=(xedges, yedges))
    return n
h = hist(d_5['X'], d_5['Y'], 50)
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.imshow(h.T)

#using gaussian_kde fucntion for density distribution

def plot_density(d1, d2, d3):
    xyz = np.vstack([d1, d2, d3])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111, projection = '3d')
    scatter = ax7.scatter(d1, d2, d3, c = density)
    fig7.colorbar(scatter, location='left')
    max_pos = xyz.T[np.argmax(density)]
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    fig7.savefig('absorber.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()
    return max_pos

# print(plot_density(x_s_1, y_s_1, z_s_1),
# plot_density(x_a_1, y_a_1, z_a_1),
# plot_density(x_s_2, y_s_2, z_s_2),
# plot_density(x_a_2, y_a_2, z_a_2),
# plot_density(x_s_3, y_s_3, z_s_3),
# plot_density(x_a_3, y_a_3, z_a_3),
# plot_density(x_s_4, y_s_4, z_s_4),
# plot_density(x_a_4, y_a_4, z_a_4),
# plot_density(x_s_5, y_s_5, z_s_5),
# plot_density(x_a_5, y_a_5, z_a_5),
# plot_density(x_s_6, y_s_6, z_s_6),
# plot_density(x_a_6, y_a_6, z_a_6),
# plot_density(x_s_7, y_s_7, z_s_7),
# plot_density(x_a_7, y_a_7, z_a_7),
# plot_density(x_s_8, y_s_8, z_s_8),
# plot_density(x_a_8, y_a_8, z_a_8))

print(plot_density(x_a, y_a, z_a))

table1 = [['Channel Number', 'Scatterer/Absorber', 'Highest Scattering Position', 'Highest Absorption Position'], 
          ['0', 'Scatterer', plot_density(x_s_1, y_s_1, z_s_1), plot_density(x_a_1, y_a_1, z_a_1)],
          ['1', 'Scatterer', plot_density(x_s_2, y_s_2, z_s_2), plot_density(x_a_2, y_a_2, z_a_2)],
          ['2', 'Scatterer', plot_density(x_s_3, y_s_3, z_s_3), plot_density(x_a_3, y_a_3, z_a_3)],
          ['3', 'Scatterer', plot_density(x_s_4, y_s_4, z_s_4), plot_density(x_a_4, y_a_4, z_a_4)],
          ['4', 'Absorber', plot_density(x_s_5, y_s_5, z_s_5), plot_density(x_a_5, y_a_5, z_a_5)],
          ['5', 'Absorber', plot_density(x_s_6, y_s_6, z_s_6), plot_density(x_a_6, y_a_6, z_a_6)],
          ['6', 'Absorber', plot_density(x_s_7, y_s_7, z_s_7), plot_density(x_a_7, y_a_7, z_a_7)],
          ['7', 'Absorber', plot_density(x_s_8, y_s_8, z_s_8), plot_density(x_a_8, y_a_8, z_a_8)]]

# print(tabulate(table1, headers='firstrow'))
with open('positionstable.txt', 'w') as f:
    f.write(tabulate(table1))
    
table2= [['Channel Number', 'Scatterer/Absorber', 'Number of Scatters', 'Number of Absorptions'],
          ['0', 'Scatterer', len(x_s_1), len(x_a_1)],
          ['1', 'Scatterer', len(x_s_2), len(x_a_2)],
          ['2', 'Scatterer', len(x_s_3), len(x_a_3)],
          ['3', 'Scatterer', len(x_s_4), len(x_a_4)],
          ['4', 'Absorber', len(x_s_5), len(x_a_5)],
          ['5', 'Absorber', len(x_s_6), len(x_a_6)],
          ['6', 'Absorber', len(x_s_7), len(x_a_7)],
          ['7', 'Absorber', len(x_s_8), len(x_a_8)]]

print(tabulate(table2, headers='firstrow'))
with open('numberofeventtypes.txt', 'w') as f:
    f.write(tabulate(table2))
    

fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
ax8.scatter(d_7['X'], d_7['Y'])