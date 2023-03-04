# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:57:54 2023

@author: CaraClarke
"""

from scipy import ndimage
import numpy as np
a1 = np.zeros((7,7), dtype=int)
a1[1:6, 2:5] = 1

b1 = ndimage.binary_erosion(a1).astype(a1.dtype)

#Erosion removes objects smaller than the structure
c1 = ndimage.binary_erosion(a1, structure=np.ones((5,5))).astype(a1.dtype)

print(a1)
print(b1)
print(c1)

a2 = np.zeros((5, 5))
a2[2, 2] = 1

b2 = ndimage.binary_dilation(a2)

c2 = ndimage.binary_dilation(a2).astype(a2.dtype)

print(a2)
print(b2)
print(c2)

