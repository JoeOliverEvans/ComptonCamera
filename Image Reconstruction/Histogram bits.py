# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:15:59 2023

@author: CaraClarke
"""
import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# Generate some test data
x = np.random.randn(10000)
y = np.random.randn(10000)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()

# 3D Heatmap in Python using matplotlib (GeeksforGeeks: https://www.geeksforgeeks.org/3d-heatmap-in-python/)
  
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
  
# creating a dummy dataset
x = np.random.randint(low=100, high=500, size=(1000,))
y = np.random.randint(low=300, high=500, size=(1000,))
z = np.random.randint(low=200, high=500, size=(1000,))
colo = [x + y + z]
  
# creating figures
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
  
# setting color bar
color_map = cm.ScalarMappable(cmap=cm.Greens_r)
color_map.set_array(colo)
  
# creating the heatmap
img = ax.scatter(x, y, z, marker='s',
                 s=200, color='green')
plt.colorbar(color_map)
  
# adding title and labels
ax.set_title("3D Heatmap")
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
  
# displaying plot
plt.show()