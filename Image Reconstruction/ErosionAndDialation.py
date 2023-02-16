#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:01:18 2023

@author: amberjones
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.draw import circle
from skimage.morphology import erosion, dilation

circ_image = np.zeros((100, 100))
circ_image[circle(50, 50, 25)] = 1

cross = np.array([[1,0,1],
                  [0,1,0],
                  [1,0,1]])


eroded_circle = erosion(circ_image, cross)
imshow(eroded_circle);


def multi_erosion(image, kernel, iterations):
    for i in range(iterations):
        image = erosion(image, kernel)
    return image
ites = [2,4,6,8,10,12,14,16,18,20]
fig, ax = plt.subplots(2, 5, figsize=(17, 5))
for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Iterations : {ites[n]}', fontsize = 16)
    new_circle = multi_erosion(circ_image, cross, ites[n])
    ax.imshow(new_circle, cmap = 'gray');
    ax.axis('off')
fig.tight_layout()

''' This link is really good at explaining erosion and dialation:
    https://towardsdatascience.com/introduction-to-image-processing-with-python-dilation-and-erosion-for-beginners-d3a0f29ad72b
    Looks like its used to isolate different parts of images
    and getting rid of unwanted shapes'''
