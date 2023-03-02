# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:31:05 2023

@author: CaraClarke
"""
#https://www.geeksforgeeks.org/how-to-load-and-save-3d-numpy-array-to-file-using-savetxt-and-loadtxt-functions/

import numpy as np


arr = np.random.rand(100, 3, 3) #gives 100 random 3x3 matrices
print(arr)

# reshaping the array from 3D to 2D
arr_reshaped = arr.reshape(arr.shape[0], -1)

# saving array
np.savetxt("file.txt", arr_reshaped)

# get file data
loaded_arr = np.loadtxt("file.txt")

# This is a 2D array - need to convert it to the original
load_original_arr = loaded_arr.reshape(
	loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])

# check
print("shape of arr: ", arr.shape)
print("shape of load_original_arr: ", load_original_arr.shape)

# check if both arrays are same or not:
if (load_original_arr == arr).all():
	print("Yes, both the arrays are same")
else:
	print("No, both the arrays are not same")
