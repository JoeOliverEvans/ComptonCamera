# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:30:28 2023

@author: JackHocking
"""

import numpy as np

def createParabola(focal_length, centre, rotation):
    t = np.linspace(-np.pi, np.pi,100)
    x_parabola = focal_length * t**2
    y_parabola = 2 * focal_length * t
    if rotation is not None:
        x_parabola, y_parabola = rotateCoordinates(x_parabola, y_parabola, rotation) 
    x_parabola = x_parabola + centre[0]
    y_parabola = y_parabola + centre[1]
    return x_parabola, y_parabola

def createCircle(radius, centre):
    theta = np.linspace(0, 2*np.pi,100)
    x_circle = radius * np.cos(theta) + centre[0]
    y_circle = radius * np.sin(theta) + centre[1]
    return x_circle, y_circle

def createEllipse(major_axis, minor_axis, centre, rotation):
    theta = np.linspace(0, 2*np.pi,100)
    x_ellipse = major_axis * np.cos(theta) 
    y_ellipse = minor_axis * np.sin(theta) 
    if rotation is not None:
        x_ellipse, y_ellipse = rotateCoordinates(x_ellipse,y_ellipse, rotation)
    x_ellipse = x_ellipse + centre[0]
    y_ellipse = y_ellipse + centre[1]
    return x_ellipse, y_ellipse

def createHyperbola(major_axis, conjugate_axis, centre, rotation):
    theta = np.linspace(0, 2*np.pi,100)
    x_hyperbola = major_axis * 1/np.cos(theta) + centre[0]
    y_hyperbola = conjugate_axis * np.tan(theta) + centre[1]
    if rotation is not None:
        x_hyperbola, y_hyperbola = rotateCoordinates(x_hyperbola, y_hyperbola, rotation)
    x_hyperbola = x_hyperbola + centre[0]
    y_hyperbola = y_hyperbola + centre[1]
    return x_hyperbola, y_hyperbola

def rotateCoordinates(x_data, y_data, rot_angle):
    x_ = x_data*np.cos(rot_angle) - y_data*np.sin(rot_angle)
    y_ = x_data*np.sin(rot_angle) + y_data*np.cos(rot_angle)
    return x_,y_