# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 08:36:56 2023

@author: AmblesideVets-NB06
"""

import numpy as np

def tilt_cone(cone, angle, axis):
    # rotation matrix
    rotation_matrix = np.identity(3)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
    # Rotate cone
    rotated_cone = np.dot(cone, rotation_matrix)
    return rotated_cone

# eg
cone = np.array([[0, 0, 0], [1, 0, 0]])
angle = np.pi / 6
axis = 'y'

tilted_cone = tilt_cone(cone, angle, axis)

print(tilted_cone)