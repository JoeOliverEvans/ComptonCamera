# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:03:42 2023

@author: CaraClarke
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

df = pd.read_parquet('15thMarchSS.parquet')

def plot_density(pos): #plotted bit is commented out and has been moved to another function - this just finds max points (saves time)
    if len(pos)==0:
        return 'No events'
    elif len(pos)==1: #takes position of single event
        return pos
    elif len(pos)==2: #takes average of positions of both of events
        return (pos.iloc[0]+pos.iloc[1])/2
    elif len(pos)==3: #takes average of positions of three of events
        return (pos.iloc[0]+pos.iloc[1]+pos.iloc[2])/2
    elif len(pos)>3:
        d1 = []
        d2 = []
        d3 = []
        for i in range(len(pos)):
            x = (pos.iloc[i])[0]
            y = (pos.iloc[i])[1]
            z = (pos.iloc[i])[2]
            d1.append([x])
            d2.append([y])
            d3.append([z])
        X = np.array(d1).flatten()
        Y = np.array(d2).flatten()
        Z = np.array(d3).flatten()
        xyz= np.vstack([X, Y, Z])
        # xyz = ', '.join(str(item) for item in xyz_)
        kde = stats.gaussian_kde(xyz)
        density = kde(xyz)
        # fig7 = plt.figure()
        # ax7 = fig7.add_subplot(111, projection = '3d')
        # scatter = ax7.scatter(X, Y, Z, c = density)
        # fig7.colorbar(scatter, location='left')
        max_pos = xyz.T[np.argmax(density)]
        # ax7.set_xlabel('X')
        # ax7.set_ylabel('Y')
        # ax7.set_zlabel('Z')
        # fig7.savefig('absorber.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        # plt.show()
        return max_pos
    
def actual_plot(pos): #actually plots density
    d1 = []
    d2 = []
    d3 = []
    for i in range(len(pos)):
        x = (pos.iloc[i])[0]
        y = (pos.iloc[i])[1]
        z = (pos.iloc[i])[2]
        d1.append([x])
        d2.append([y])
        d3.append([z])
    X = np.array(d1).flatten()
    Y = np.array(d2).flatten()
    Z = np.array(d3).flatten()
    xyz= np.vstack([X, Y, Z])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111, projection = '3d')
    scatter = ax7.scatter(X, Y, Z, c = density)
    fig7.colorbar(scatter, location='left')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    fig7.savefig('3to7ss.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()
    

#split into scattering and absorption detctors
#scattering events
d_1s = df.query('`scatter channel` == 0') #events that scatter in channel 0 etc
d_2s = df.query('`scatter channel` == 1')
d_3s = df.query('`scatter channel` == 2')
d_4s = df.query('`scatter channel` == 3')
d_5s = df.query('`scatter channel` == 4')
d_6s = df.query('`scatter channel` == 5')
d_7s = df.query('`scatter channel` == 6')
d_8s = df.query('`scatter channel` == 7')

#specific scatter locations in each detector
d_1ssp = d_1s['scatter locations'] #specific scatter locations of photons that have scattered in channel 0
d_2ssp = d_2s['scatter locations']
d_3ssp = d_3s['scatter locations']
d_4ssp = d_4s['scatter locations']
d_5ssp = d_5s['scatter locations']
d_6ssp = d_6s['scatter locations']
d_7ssp = d_7s['scatter locations']
d_8ssp = d_8s['scatter locations']

# def getxyz(pos):
#     x = []
#     y = []
#     z = []
#     for i in range(len(pos)):
#         d1 = (pos.iloc[i])[0]
#         d2 = (pos.iloc[i])[1]
#         d3 = (pos.iloc[i])[2]
#         x.append([d1])
#         y.append([d2])
#         z.append([d3])
#     return x, y, z

# x, y, z = getxyz(df['scatter locations'])
# X = np.array(x).flatten()
# Y = np.array(y).flatten()
# Z = np.array(z).flatten()
# fig7 = plt.figure()
# ax7 = fig7.add_subplot(111, projection = '3d')
# ax7.scatter(X, Y, Z)

# list_of_numbers = np.stack([x, y, z])

# result = ', '.join(str(item) for item in list_of_numbers)
# print(list_of_numbers)
# print(result)


#specific absorption positions from each scattering detector
d_1sap = d_1s['absorption locations'] #specific absorption positions of photons that have scattered in channel 0
d_2sap = d_2s['absorption locations']
d_3sap = d_3s['absorption locations']
d_4sap = d_4s['absorption locations']
d_5sap = d_5s['absorption locations']
d_6sap = d_6s['absorption locations']
d_7sap = d_7s['absorption locations']
d_8sap = d_8s['absorption locations']

#absorption events
d_1a = df.query('`absorption channel` == 0') #events that absorb in channel 0 etc
d_2a = df.query('`absorption channel` == 1')
d_3a = df.query('`absorption channel` == 2')
d_4a = df.query('`absorption channel` == 3')
d_5a = df.query('`absorption channel` == 4')
d_6a = df.query('`absorption channel` == 5')
d_7a = df.query('`absorption channel` == 6')
d_8a = df.query('`absorption channel` == 7')

#specific scattering positions from each absorption detector
d_1asp = d_1a['scatter locations'] #e.g specific scattering positions of events that absorb in channel 0
d_2asp = d_2a['scatter locations']
d_3asp = d_3a['scatter locations']
d_4asp = d_4a['scatter locations']
d_5asp = d_5a['scatter locations']
d_6asp = d_6a['scatter locations']
d_7asp = d_7a['scatter locations']
d_8asp = d_8a['scatter locations']

#specific absorption posiitons from each absorption detector
d_1aap = d_1a['absorption locations'] #e.g. specific absorption posiitons of events that absorb in channel 0
d_2aap = d_2a['absorption locations']
d_3aap = d_3a['absorption locations']
d_4aap = d_4a['absorption locations']
d_5aap = d_5a['absorption locations']
d_6aap = d_6a['absorption locations']
d_7aap = d_7a['absorption locations']
d_8aap = d_8a['absorption locations']

#channel 0 
d_1s_1a = d_1s.query('`absorption channel` == 0') #events that scatter in channel 0 and absorb in channel 0 etc
d_1s_2a = d_1s.query('`absorption channel` == 1')
d_1s_3a = d_1s.query('`absorption channel` == 2')
d_1s_4a = d_1s.query('`absorption channel` == 3')
d_1s_5a = d_1s.query('`absorption channel` == 4')
d_1s_6a = d_1s.query('`absorption channel` == 5')
d_1s_7a = d_1s.query('`absorption channel` == 6')
d_1s_8a = d_1s.query('`absorption channel` == 7')


#channel 1
d_2s_1a = d_2s.query('`absorption channel` == 0') #events that scatter in channel 1 and absorb in channel 0 etc
d_2s_2a = d_2s.query('`absorption channel` == 1')
d_2s_3a = d_2s.query('`absorption channel` == 2')
d_2s_4a = d_2s.query('`absorption channel` == 3')
d_2s_5a = d_2s.query('`absorption channel` == 4')
d_2s_6a = d_2s.query('`absorption channel` == 5')
d_2s_7a = d_2s.query('`absorption channel` == 6')
d_2s_8a = d_2s.query('`absorption channel` == 7')

#channel 2
d_3s_1a = d_3s.query('`absorption channel` == 0') #events that scatter in channel 2 and absorb in channel 0 etc
d_3s_2a = d_3s.query('`absorption channel` == 1')
d_3s_3a = d_3s.query('`absorption channel` == 2')
d_3s_4a = d_3s.query('`absorption channel` == 3')
d_3s_5a = d_3s.query('`absorption channel` == 4')
d_3s_6a = d_3s.query('`absorption channel` == 5')
d_3s_7a = d_3s.query('`absorption channel` == 6')
d_3s_8a = d_3s.query('`absorption channel` == 7')

#channel 3
d_4s_1a = d_4s.query('`absorption channel` == 0') #events that scatter in channel 3 and absorb in channel 0 etc
d_4s_2a = d_4s.query('`absorption channel` == 1')
d_4s_3a = d_4s.query('`absorption channel` == 2')
d_4s_4a = d_4s.query('`absorption channel` == 3')
d_4s_5a = d_4s.query('`absorption channel` == 4')
d_4s_6a = d_4s.query('`absorption channel` == 5')
d_4s_7a = d_4s.query('`absorption channel` == 6')
d_4s_8a = d_4s.query('`absorption channel` == 7')


#channel 4 
d_5s_1a = d_5s.query('`absorption channel` == 0') #events that scatter in channel 4 and absorb in channel 0 etc
d_5s_2a = d_5s.query('`absorption channel` == 1')
d_5s_3a = d_5s.query('`absorption channel` == 2')
d_5s_4a = d_5s.query('`absorption channel` == 3')
d_5s_5a = d_5s.query('`absorption channel` == 4')
d_5s_6a = d_5s.query('`absorption channel` == 5')
d_5s_7a = d_5s.query('`absorption channel` == 6')
d_5s_8a = d_5s.query('`absorption channel` == 7')

#channel 5
d_6s_1a = d_6s.query('`absorption channel` == 0') #events that scatter in channel 5 and absorb in channel 0 etc
d_6s_2a = d_6s.query('`absorption channel` == 1')
d_6s_3a = d_6s.query('`absorption channel` == 2')
d_6s_4a = d_6s.query('`absorption channel` == 3')
d_6s_5a = d_6s.query('`absorption channel` == 4')
d_6s_6a = d_6s.query('`absorption channel` == 5')
d_6s_7a = d_6s.query('`absorption channel` == 6')
d_6s_8a = d_6s.query('`absorption channel` == 7')

#channel 6
d_7s_1a = d_7s.query('`absorption channel` == 0') #events that scatter in channel 6 and absorb in channel 0 etc
d_7s_2a = d_7s.query('`absorption channel` == 1')
d_7s_3a = d_7s.query('`absorption channel` == 2')
d_7s_4a = d_7s.query('`absorption channel` == 3')
d_7s_5a = d_7s.query('`absorption channel` == 4')
d_7s_6a = d_7s.query('`absorption channel` == 5')
d_7s_7a = d_7s.query('`absorption channel` == 6')
d_7s_8a = d_7s.query('`absorption channel` == 7')

#channel 7
d_8s_1a = d_8s.query('`absorption channel` == 0') #events that scatter in channel 7 and absorb in channel 0 etc
d_8s_2a = d_8s.query('`absorption channel` == 1')
d_8s_3a = d_8s.query('`absorption channel` == 2')
d_8s_4a = d_8s.query('`absorption channel` == 3')
d_8s_5a = d_8s.query('`absorption channel` == 4')
d_8s_6a = d_8s.query('`absorption channel` == 5')
d_8s_7a = d_8s.query('`absorption channel` == 6')
d_8s_8a = d_8s.query('`absorption channel` == 7')

#table for frequency of event sequence
table1 = [['Scattering Detector 1', 'Scattering Detector 2', 'Number of Events', 'Highest Scattering Point in 1', 'Highest Scattering Point in 2'], 
          ['0', '0', len(d_1s_1a), plot_density(d_1s_1a['scatter locations']), plot_density(d_1s_1a['absorption locations'])],
          ['0', '1', len(d_1s_2a), plot_density(d_1s_2a['scatter locations']), plot_density(d_1s_2a['absorption locations'])],
          ['0', '2', len(d_1s_3a), plot_density(d_1s_3a['scatter locations']), plot_density(d_1s_3a['absorption locations'])],
          ['0', '3', len(d_1s_4a), plot_density(d_1s_4a['scatter locations']), plot_density(d_1s_4a['absorption locations'])],
          ['0', '4', len(d_1s_5a), plot_density(d_1s_5a['scatter locations']), plot_density(d_1s_5a['absorption locations'])],
          ['0', '5', len(d_1s_6a), plot_density(d_1s_6a['scatter locations']), plot_density(d_1s_6a['absorption locations'])],
          ['0', '6', len(d_1s_7a), plot_density(d_1s_7a['scatter locations']), plot_density(d_1s_7a['absorption locations'])],
          ['0', '7', len(d_1s_8a), plot_density(d_1s_8a['scatter locations']), plot_density(d_1s_8a['absorption locations'])],
          ['1', '0', len(d_2s_1a), plot_density(d_2s_1a['scatter locations']), plot_density(d_2s_1a['absorption locations'])],
          ['1', '1', len(d_2s_2a), plot_density(d_2s_2a['scatter locations']), plot_density(d_2s_2a['absorption locations'])],
          ['1', '2', len(d_2s_3a), plot_density(d_2s_3a['scatter locations']), plot_density(d_2s_3a['absorption locations'])],
          ['1', '3', len(d_2s_4a), plot_density(d_2s_4a['scatter locations']), plot_density(d_2s_4a['absorption locations'])],
          ['1', '4', len(d_2s_5a), plot_density(d_2s_5a['scatter locations']), plot_density(d_2s_5a['absorption locations'])],
          ['1', '5', len(d_2s_6a), plot_density(d_2s_6a['scatter locations']), plot_density(d_2s_6a['absorption locations'])],
          ['1', '6', len(d_2s_7a), plot_density(d_2s_7a['scatter locations']), plot_density(d_2s_7a['absorption locations'])],
          ['1', '7', len(d_2s_8a), plot_density(d_2s_8a['scatter locations']), plot_density(d_2s_8a['absorption locations'])],
          ['2', '0', len(d_3s_1a), plot_density(d_3s_1a['scatter locations']), plot_density(d_3s_1a['absorption locations'])],
          ['2', '1', len(d_3s_2a), plot_density(d_3s_2a['scatter locations']), plot_density(d_3s_2a['absorption locations'])],
          ['2', '2', len(d_3s_3a), plot_density(d_3s_3a['scatter locations']), plot_density(d_3s_3a['absorption locations'])],
          ['2', '3', len(d_3s_4a), plot_density(d_3s_4a['scatter locations']), plot_density(d_3s_4a['absorption locations'])],
          ['2', '4', len(d_3s_5a), plot_density(d_3s_5a['scatter locations']), plot_density(d_3s_5a['absorption locations'])],
          ['2', '5', len(d_3s_6a), plot_density(d_3s_6a['scatter locations']), plot_density(d_3s_6a['absorption locations'])],
          ['2', '6', len(d_3s_7a), plot_density(d_3s_7a['scatter locations']), plot_density(d_3s_7a['absorption locations'])],
          ['2', '7', len(d_3s_8a), plot_density(d_3s_8a['scatter locations']), plot_density(d_3s_8a['absorption locations'])],
          ['3', '0', len(d_4s_1a), plot_density(d_4s_1a['scatter locations']), plot_density(d_4s_1a['absorption locations'])],
          ['3', '1', len(d_4s_2a), plot_density(d_4s_2a['scatter locations']), plot_density(d_4s_2a['absorption locations'])],
          ['3', '2', len(d_4s_3a), plot_density(d_4s_3a['scatter locations']), plot_density(d_4s_3a['absorption locations'])],
          ['3', '3', len(d_4s_4a), plot_density(d_4s_4a['scatter locations']), plot_density(d_4s_4a['absorption locations'])],
          ['3', '4', len(d_4s_5a), plot_density(d_4s_5a['scatter locations']), plot_density(d_4s_5a['absorption locations'])],
          ['3', '5', len(d_4s_6a), plot_density(d_4s_6a['scatter locations']), plot_density(d_4s_6a['absorption locations'])],
          ['3', '6', len(d_4s_7a), plot_density(d_4s_7a['scatter locations']), plot_density(d_4s_7a['absorption locations'])],
          ['3', '7', len(d_4s_8a), plot_density(d_4s_8a['scatter locations']), plot_density(d_4s_8a['absorption locations'])],
          ['4', '0', len(d_5s_1a), plot_density(d_5s_1a['scatter locations']), plot_density(d_5s_1a['absorption locations'])],
          ['4', '1', len(d_5s_2a), plot_density(d_5s_2a['scatter locations']), plot_density(d_5s_2a['absorption locations'])],
          ['4', '2', len(d_5s_3a), plot_density(d_5s_3a['scatter locations']), plot_density(d_5s_3a['absorption locations'])],
          ['4', '3', len(d_5s_4a), plot_density(d_5s_4a['scatter locations']), plot_density(d_5s_4a['absorption locations'])],
          ['4', '4', len(d_5s_5a), plot_density(d_5s_5a['scatter locations']), plot_density(d_5s_5a['absorption locations'])],
          ['4', '5', len(d_5s_6a), plot_density(d_5s_6a['scatter locations']), plot_density(d_5s_6a['absorption locations'])],
          ['4', '6', len(d_5s_7a), plot_density(d_5s_7a['scatter locations']), plot_density(d_5s_7a['absorption locations'])],
          ['4', '7', len(d_5s_8a), plot_density(d_5s_8a['scatter locations']), plot_density(d_5s_8a['absorption locations'])],
          ['5', '0', len(d_6s_1a), plot_density(d_6s_1a['scatter locations']), plot_density(d_6s_1a['absorption locations'])],
          ['5', '1', len(d_6s_2a), plot_density(d_6s_2a['scatter locations']), plot_density(d_6s_2a['absorption locations'])],
          ['5', '2', len(d_6s_3a), plot_density(d_6s_3a['scatter locations']), plot_density(d_6s_3a['absorption locations'])],
          ['5', '3', len(d_6s_4a), plot_density(d_6s_4a['scatter locations']), plot_density(d_6s_4a['absorption locations'])],
          ['5', '4', len(d_6s_5a), plot_density(d_6s_5a['scatter locations']), plot_density(d_6s_5a['absorption locations'])],
          ['5', '5', len(d_6s_6a), plot_density(d_6s_6a['scatter locations']), plot_density(d_6s_6a['absorption locations'])],
          ['5', '6', len(d_6s_7a), plot_density(d_6s_7a['scatter locations']), plot_density(d_6s_7a['absorption locations'])],
          ['5', '7', len(d_6s_8a), plot_density(d_6s_8a['scatter locations']), plot_density(d_6s_8a['absorption locations'])],
          ['6', '0', len(d_7s_1a), plot_density(d_7s_1a['scatter locations']), plot_density(d_7s_1a['absorption locations'])],
          ['6', '1', len(d_7s_2a), plot_density(d_7s_2a['scatter locations']), plot_density(d_7s_2a['absorption locations'])],
          ['6', '2', len(d_7s_3a), plot_density(d_7s_3a['scatter locations']), plot_density(d_7s_3a['absorption locations'])],
          ['6', '3', len(d_7s_4a), plot_density(d_7s_4a['scatter locations']), plot_density(d_7s_4a['absorption locations'])],
          ['6', '4', len(d_7s_5a), plot_density(d_7s_5a['scatter locations']), plot_density(d_7s_5a['absorption locations'])],
          ['6', '5', len(d_7s_6a), plot_density(d_7s_6a['scatter locations']), plot_density(d_7s_6a['absorption locations'])],
          ['6', '6', len(d_7s_7a), plot_density(d_7s_7a['scatter locations']), plot_density(d_7s_7a['absorption locations'])],
          ['6', '7', len(d_7s_8a), plot_density(d_7s_8a['scatter locations']), plot_density(d_7s_8a['absorption locations'])],
          ['7', '0', len(d_8s_1a), plot_density(d_8s_1a['scatter locations']), plot_density(d_8s_1a['absorption locations'])],
          ['7', '1', len(d_8s_2a), plot_density(d_8s_2a['scatter locations']), plot_density(d_8s_2a['absorption locations'])],
          ['7', '2', len(d_8s_3a), plot_density(d_8s_3a['scatter locations']), plot_density(d_8s_3a['absorption locations'])],
          ['7', '3', len(d_8s_4a), plot_density(d_8s_4a['scatter locations']), plot_density(d_8s_4a['absorption locations'])],
          ['7', '4', len(d_8s_5a), plot_density(d_8s_5a['scatter locations']), plot_density(d_8s_5a['absorption locations'])],
          ['7', '5', len(d_8s_6a), plot_density(d_8s_6a['scatter locations']), plot_density(d_8s_6a['absorption locations'])],
          ['7', '6', len(d_8s_7a), plot_density(d_8s_7a['scatter locations']), plot_density(d_8s_7a['absorption locations'])],
          ['7', '7', len(d_8s_8a), plot_density(d_8s_8a['scatter locations']), plot_density(d_8s_8a['absorption locations'])]]

# print(tabulate(table1, headers='firstrow'))
with open('tableofeventdistributions(SS).txt', 'w') as f:
    f.write(tabulate(table1))
# content2=tabulate(table1, tablefmt="tsv")
# text_file=open("tableofeventdistributions.csv","w")
# text_file.write(content2)
# text_file.close()
