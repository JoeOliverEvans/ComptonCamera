# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:47:41 2023

@author: CaraClarke
"""

import numpy as np

def rotation_matrix(vec1, vec2):
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

#example
vect1 = [0, 0, 1]
vect2 = [1, 1, 1]
Rot = rotation_matrix(vect1, vect2)
new = Rot.dot(vect1)
print(Rot)
print(new)
