# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:15:30 2023

@author: CaraCLarke
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

#get data
df = pd.read_csv('result.csv', sep=';')
df.to_parquet('result.parquet')
df2 = pd.read_parquet('result.parquet') 
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
# ax4.view_init(0, -90)
# x_centers = (edges[0][1:] + edges[0][:-1]) / 2
# y_centers = (edges[1][1:] + edges[1][:-1]) / 2
# z_centers = (edges[2][1:] + edges[2][:-1]) / 2
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')

# # Create a meshgrid of the bin centers for each axis
# x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

# # Flatten the histogram and the meshgrid
# hist_flat = hist_.flatten()
# x_flat = (X*50)/184
# y_flat = (Y*50)/184
# z_flat = (Z*50)/184

# # Create the 3D density heat map using the scatter() function
# ax4.scatter(x_flat, y_flat, z_flat, c=hist_flat, marker='o')

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
h = hist(d_5['X'], d_5['Y'], 20)
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.imshow(h.T)

#using gaussian_kde fucntion for density distribution
xyz = np.vstack([d_1['X'],d_1['Y'],d_1['Z']])
kde = stats.gaussian_kde(xyz)
density = kde(xyz)
fig7 = plt.figure()
ax7 = fig7.add_subplot(111, projection = '3d')
scatter = ax7.scatter(d_1['X'], d_1['Y'], d_1['Z'], c = density)
fig7.colorbar(scatter)
max_pos = xyz.T[np.argmax(density)]

# def plot4d(data):
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(projection="3d")
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     mask = data > 0.01
#     idx = np.arange(int(np.prod(data.shape)))
#     x, y, z = np.unravel_index(idx, data.shape)
#     ax.scatter(x, y, z, c=data.flatten(), s=10.0 * mask, edgecolor="face", alpha=0.2, marker="o", cmap="magma", linewidth=0)
#     plt.tight_layout()
#     plt.savefig("test_scatter_4d.png", dpi=250)
#     plt.close(fig)


# if __name__ == "__main__":
#     # X = np.arange(-10, 10, 0.5)
#     # Y = np.arange(-10, 10, 0.5)
#     # Z = np.arange(-10, 10, 0.5)
#     X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")
#     density_matrix = np.sin(np.sqrt(X**2 + Y**2 + Z**2))
#     plot4d(density_matrix)

