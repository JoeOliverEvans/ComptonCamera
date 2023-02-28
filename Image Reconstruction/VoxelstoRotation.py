# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:22:22 2023

@author: CaraClarke (originally amber and joe's code)
"""

import numpy as np
import matplotlib.pyplot as plt

N = 200
theta_c = (np.pi/4)-0.001
voxel_length = 1
R_max = 4
# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(0, 1, N)

theta = np.linspace(0, 2*np.pi, N)

R = np.arange(0, R_max, voxel_length/4)
X = []
Y = []
Z = []
for r in R:
    Theta_size = 2 * np.arctan((voxel_length/4)/(np.sin(theta_c) * r))
    print(Theta_size)
    Theta = np.linspace(0, 2 * np.pi, int(np.abs(2*np.pi//Theta_size) + 1))
    for x in r*np.sin(Theta):
        X.append(x)
    for y in r*np.cos(Theta):
        Y.append(y)
        Z.append(np.cos(theta_c)*r)
print(X)
print(np.shape(X))
print(Y)
print(np.shape(Y))
print(Z)
print(np.shape(Z))

#Plot the surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, c=Z, cmap=plt.cm.twilight_shifted)


def RotationMatrix_x(theta):
    matrix_x = np.matrix([[1, 0, 0],
                 [0, np.cos(theta), -np.sin(theta)],
                 [0, np.sin(theta), np.cos(theta)]])
    return matrix_x
                         
def RotationMatrix_y(theta):    
    matrix_y = np.matrix([[np.cos(theta), 0, np.sin(theta)],
                 [0, 1, 0],
                 [-np.sin(theta), 0, np.cos(theta)]])
    return matrix_y
  
def RotationMatrix_z(theta): 
    matrix_z = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
    return matrix_z
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


pos = np.dstack([X, Y, Z])
# print(pos)




#example
theta1 = 0
theta2 = -np.pi/2
theta3 = 0

Rot = RotationMatrix_x(theta1)*RotationMatrix_y(theta2)*RotationMatrix_z(theta3)
print(Rot)

v = np.array([[1],[1],[1]])
print(v)

new_v = np.dot(Rot, v)
print(new_v)

X_2 = []
Y_2 = []
Z_2 = []

#moredataexample
for i in range(N):
    x=X[i]
    y=Y[i]
    z=Z[i]
    vect = np.array([[x], [y], [z]])
    # print(vect)
    new_vect = np.dot(Rot, vect)
    # print(new_vect)
    new_x = new_vect[0]
    new_y = new_vect[1]
    new_z = new_vect[2]
    # print(np.array([[new_x], [new_y], [new_z]]))
    X_2.append(new_x)
    Y_2.append(new_y)
    Z_2.append(new_z)

ax.scatter(X_2, Y_2, Z_2)#, c=Z_2, cmap=plt.cm.twilight_shifted)