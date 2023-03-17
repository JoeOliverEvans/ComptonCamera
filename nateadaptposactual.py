#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:39:18 2023

@author: amberjones
"""
import pandas as pd
import numpy as np
import scipy.constants as constants

data=pd.read_csv(r'/Users/amberjones/Documents/IRdata/adatactual', delimiter=';', header=0) #download the file and put the path here
print(data.iloc[1,1], data.iloc[2,1], data.iloc[3,1])
tol=1 #coincidence window in picoseconds/monte carlo timestamp units
coincidences_index=[] #stores the index of coincidences
detector_number=8 #number of detectors
source=np.array([0,0,0])
detector_pos=[np.array([-3.2, 3.5, 39.6]),np.array([2.3, -2.2, 41.3]),np.array([-3.4, -5.0, 40.8]),np.array([10.6, 2.7, 39.2]),np.array([-13.4, 0, 80.6]),np.array([0, 13.4, 80.6]),np.array([13.4, 0, 80.6]),np.array([0, -13.4, 80.6])]
m=-400 #bad angle

angles=[]
for i in detector_pos:
    line=[]
    for j in detector_pos:
        a=i-source
        b=j-i
        if -np.pi<np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))<np.pi:
            theta=np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

        else:
            theta=m
        line.append(theta)
    angles.append(line)


min_angles=list(np.asarray(angles)-0.4) #matrix of minimum angles to go from one detector to the other in radians
max_angles=list(np.asarray(angles)+0.4) #matrix of maxiumum angles to go from one detector to the other in radians
Ei=0.662 #initial energy in MeV
gap=0.01 #tolerance for absorption in MeV


for i in range(len(data.iloc[:,2])-1): #for all the data points
    if np.abs(data.iloc[i,2]-data.iloc[i+1,2]) < tol: #is the time diference between points less than the coincidence window
        coincidences_index.append(i)


scattera_energies=[] #energies of scatters followed by absorptions
scattera_locations=[] #locations of scatters followed by absorptions
absorption_energies=[] #energies of absorptions
absorption_locations=[] #locations of absorptions
scatters_energies=[] #energies of scatters followed by scatters
scatters_locations=[] #locations of scatters followed by scatters
scatter2_energies=[] #energies of second scatters
scatter2_locations=[] #locations of second scatters
lost=[]

electron_mass = (constants.electron_mass * constants.c ** 2)/ (constants.electron_volt * 10 ** 6)  # in MeV

def CalculateScatterAngle(initial_energy, final_energy):
    """
    :param final_energy:
    :param initial_energy:
    :return: Compton Scattering Angle in radians
    """
    if -1<1 - (electron_mass * ((initial_energy - final_energy) / (initial_energy * final_energy)))<1: #if arccos can be calculated
        return np.arccos(
        1 - (electron_mass * ((initial_energy - final_energy) / (initial_energy * final_energy))))
    else: #if arccos not calculatable then impossible angle
        return(2*m)

cal_grad=[1,1,1,1,1,1,1,1] #list of gradients of the calibration for each detector
cal_intercept=[0,0,0,0,0,0,0,0] #list of intercepts of the calibration for each detector

for index_index, coincidence_index in enumerate(coincidences_index): #for all the coincidences
    for i in range(detector_number):
        for j in range(detector_number):
            if data.iloc[coincidence_index,1]==i: #if the coincidence was registered in detector i
                if data.iloc[coincidence_index+1,1]==j: #if the second detector involved was detector j
                    E0=(data.iloc[coincidence_index,3]-cal_intercept[i])/cal_grad[i]
                    E1=(data.iloc[coincidence_index+1,3]-cal_intercept[j])/cal_grad[j]
                    angle_scatterer_1=CalculateScatterAngle(0.662,0.662-E0) #angle of scatter if scatter from i to j
                    angle_scatterer_2=CalculateScatterAngle(0.662,0.662-E1) #angle of scatter if scatter from j to i
                    if min_angles[i][j]<angle_scatterer_1<max_angles[i][j]: #if the angle from i to j is within the geometrically allowed values
                        if np.abs(E0+E1-Ei)<gap: #if scatter-absorption
                            scattera_energies.append(E0) #add scatter energy to list
                            scattera_locations.append([data.iloc[coincidence_index,6], data.iloc[coincidence_index,7], data.iloc[coincidence_index,8]])#add scatter location to list
                            absorption_energies.append(E1) #add absorption energy to list
                            absorption_locations.append([data.iloc[coincidence_index + 1,6], data.iloc[coincidence_index + 1,7], data.iloc[coincidence_index + 1,8]]) #add absorption locator to list
                        else: #if scatter-scatter
                            scatters_energies.append(E0) #add scatter 1 energy to list
                            scatters_locations.append([data.iloc[coincidence_index,6], data.iloc[coincidence_index,7], data.iloc[coincidence_index,8]]) #add scatter 1 location to list
                            scatter2_energies.append(E1) #add scatter 2 energy to list
                            scatter2_locations.append([data.iloc[coincidence_index + 1,6], data.iloc[coincidence_index + 1,7], data.iloc[coincidence_index + 1,8]]) #add scatter 2 locator to list
                    elif min_angles[j][i]<angle_scatterer_2<max_angles[j][i]: #if the angle from j to i is within the geometrically allowed values
                        if np.abs(E0+E1-Ei)<gap:
                            scattera_energies.append(E1) #add scatter energy to list
                            scattera_locations.append([data.iloc[coincidence_index + 1,6], data.iloc[coincidence_index + 1,7], data.iloc[coincidence_index + 1,8]]) #add scatter location to list
                            absorption_energies.append(E0) #add absorption energy to list
                            absorption_locations.append([data.iloc[coincidence_index,6], data.iloc[coincidence_index,7], data.iloc[coincidence_index,8]]) #add absorption location to list
                        else:
                            scatters_energies.append(E1) #add scatter 1 energy to list
                            scatters_locations.append([data.iloc[coincidence_index + 1,6], data.iloc[coincidence_index + 1,7], data.iloc[coincidence_index + 1,8]]) #add scatter 1 location to list
                            scatter2_energies.append(E0) #add scatter 2 energy to list
                            scatter2_locations.append([data.iloc[coincidence_index,6], data.iloc[coincidence_index,7], data.iloc[coincidence_index,8]]) #add scatter 2 location to list
                    else:
                        lost.append(coincidence_index)


# dictionary of lists
dict = {'scatter energy': scattera_energies, 'scatter locations': scattera_locations, 'absorption energies': absorption_energies, 'absorption locations': absorption_locations}

df = pd.DataFrame(dict)

print(df.iloc[:,1])
# saving the dataframe
df.to_parquet('/Users/amberjones/Documents/IRdata/dataprocessed')


