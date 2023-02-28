# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:46:12 2023

@author: CaraClarke
"""

import numpy as np
import matplotlib.pyplot as plt


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

#example
N=10 #number of points
x = np.arange(0,10,1) #coordinates of some points along the z axis
y = np.zeros(10)
z = np.zeros(10)
ax.scatter(x, y, z); #placing particles on 3d graph

pos = np.dstack([x, y, z])
# print(pos)




#example
theta1 = 0
theta2 = np.pi/4
theta3 = 0

Rot = RotationMatrix_x(theta1)*RotationMatrix_y(theta2)*RotationMatrix_z(theta3)
print(Rot)

v = np.array([[1],[1],[1]])
print(v)

new_v = np.dot(Rot, v)
print(new_v)

X = []
Y = []
Z = []

#moredataexample
for i in range(N):
    x2=x[i]
    y2=y[i]
    z2=z[i]
    vect = np.array([[x2], [y2], [z2]])
    print(vect)
    new_vect = np.dot(Rot, vect)
    print(new_vect)
    new_x = new_vect[0]
    new_y = new_vect[1]
    new_z = new_vect[2]
    print(np.array([[new_x], [new_y], [new_z]]))
    X.append(new_x)
    Y.append(new_y)
    Z.append(new_z)

ax.scatter(X, Y, Z)
