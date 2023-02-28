import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

N = 50
theta_c = 0.65
theta_err = theta_c * 0.05
theta_min, theta_max = theta_c - theta_err, theta_c + theta_err
theta_arr = np.linspace(theta_min, theta_max, N)
voxel_length = 1
R_max = 4

# Generate gaussian to find weighting
x = np.linspace(-2, 2, N)
W_arr = norm.pdf(x) * np.sqrt(2*np.pi)

# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(0, 1, N)

theta = np.linspace(0, 2*np.pi, N)

R = np.arange(0, R_max, voxel_length/4)
X = []
Y = []
Z = []
W = [] #point weighting based on gaussian distribution
for i in range(N):
    for r in R:
        Theta_size = 2 * np.arctan((voxel_length/4)/(np.sin(theta_arr[i]) * r))
        Theta = np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//Theta_size) + 1))
        for x in r*np.sin(Theta):
            X.append(x)
            W.append(W_arr[i])
        for y in r*np.cos(Theta):
            Y.append(y)
            Z.append(np.cos(theta_c)*r)

print( np.shape(W))
# Plot the surface
ax.scatter(X, Y, Z, c=W, cmap=plt.cm.plasma_r)

ax.set_zlim(0, 2)
plt.show()
