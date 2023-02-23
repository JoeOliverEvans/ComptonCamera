import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import matplotlib; matplotlib.use("TkAgg")
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from mayavi import mlab
import warnings


warnings.filterwarnings("ignore", message="invalid value encountered")

electron_mass = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  # in keV


def CalculateScatterAngle(initial_energy, final_energy):
    """
    :param final_energy:
    :param initial_energy:
    :return: Compton Scattering Angle in radians
    """
    return np.arccos(
        1 - (electron_mass * ((initial_energy - final_energy) / (initial_energy * final_energy))))


class DetectionPair:
    def __init__(self, scatter_position, absorption_position, initial_energy, absorption_energy, angle):
        """
        :param scatter_position: Coordinates of scatter
        :param absorption_position: Coordinates of absorption
        :param initial_energy:
        :param absorption_energy:
        """
        self.scatterPosition = np.float64(scatter_position)
        self.absorptionPosition = np.float64(absorption_position)
        self.lineVector = np.array((np.array(self.scatterPosition) - np.array(self.absorptionPosition)) / np.linalg.norm(np.array(self.scatterPosition) - np.array(self.absorptionPosition)), dtype=np.float64)
        self.absorptionEnergy = np.float64(absorption_energy)
        self.scatterAngle = np.float64(angle)  # CalculateScatterAngle(initial_energy, absorption_energy)


def calculate_cone_z(pair_of_detections, x, y):
    """
    :param pair_of_detections: the pair of detections that produce the cone
    :param x:
    :param y:
    :return: z1 and z2 which contain the z value, there are two as it is the solution to a quadratic
    """
    a = pair_of_detections.lineVector[0]
    b = pair_of_detections.lineVector[1]
    c = pair_of_detections.lineVector[2]
    t = pair_of_detections.scatterAngle
    g = np.cos(t) * (a ** 2 + b ** 2 + c ** 2) ** (1 / 2)
    d = (c ** 2 / g ** 2) - 1
    e = (2 * a * x * c + b * y * c) / (g ** 2)
    f = (((a * x) ** 2 + (b * y) ** 2 + 2 * a * x * b * y) / (g ** 2)) - x ** 2 - y ** 2
    h = e ** 2 - 4 * d * f
    z1 = ((-e - np.sqrt(h)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    z2 = ((-e + np.sqrt(h)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    return z1, z2


def plot_3d(view_only_intersections=True, min_number_of_intersections=2):
    """
    Function to view a 3D plot of cones, can toggle to see only intersections
    :param min_number_of_intersections:
    :param view_only_intersections: True or False
    :return:
    """
    # set the colors of each object
    colors = np.empty(voxel_cube.shape, dtype=object)
    cones = np.where(voxel_cube < 1, voxel_cube, False)

    if view_only_intersections:
        intersections = voxel_cube >= np.max(voxel_cube)
        # intersections = np.where(voxel_cube > 1, voxel_cube, True)
        intersections = np.array(intersections, dtype=bool)
        colors[intersections] = 'green'

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(intersections, facecolors=colors, edgecolor='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    else:
        colors[voxel_cube == 1] = 'blue'
        colors[voxel_cube > 1] = 'green'

        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.voxels(voxel_cube, facecolors=colors, edgecolor='k')
        plt.show()


def mayavi_plot_3d(voxel_cube_maya, view_only_intersections=True, min_intersections=-1):
    if min_intersections == -1:
        min_intersections = np.max(voxel_cube_maya)
    max_intersections_arguments = np.array(np.argwhere(voxel_cube >0))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    #np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c*voxel_length,v*voxel_length,b*voxel_length, voxel_cube[c, v, b], mode='cube', color=(0, 1, 0), scale_mode='none', scale_factor=voxel_length)
    '''max_intersections_arguments = np.array(np.argwhere(voxel_cube > 1))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    #np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c*voxel_length,v*voxel_length,b*voxel_length, voxel_cube[c, v, b], mode='cube', scale_mode='none', scale_factor=voxel_length, opacity=0.1, colormap='autumn')'''
    mlab.axes(xlabel='x', ylabel='y', zlabel='z', extent=(0, 40, 0, 40, 0, 40), nb_labels=10)
    mlab.show()


def calculate_voxel_cone_cube(arg):
    imaging_area, voxel_length, voxels_per_side, checks_per_side, pair_of_detections = arg[0], arg[1], arg[2], arg[3], arg[4]
    x_values = np.array(np.tile(np.arange(0, imaging_area[0], voxel_length / checks_per_side) - pair_of_detections.scatterPosition[0],
                       (voxels_per_side[1] * checks_per_side, 1)), dtype=np.float64)
    y_values = np.array(np.tile(
        np.array(
            [np.arange(0, imaging_area[1], voxel_length / checks_per_side) - pair_of_detections.scatterPosition[1]]).transpose(),
        (1, voxels_per_side[0] * checks_per_side)), dtype=np.float64)
    z1, z2 = calculate_cone_z(pair_of_detections, x_values, y_values)
    voxel_cube_cone = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)
    for (x1, y1) in np.argwhere((0 <= z1) & (z1 < imaging_area[2])):
        voxel_cube_cone[y1 // checks_per_side, x1 // checks_per_side, int(z1[x1, y1] // voxel_length)] = 1
    for (x2, y2) in np.argwhere((0 <= z2) & (z2 < imaging_area[2])):
        voxel_cube_cone[y2 // checks_per_side, x2 // checks_per_side, int(z2[x2, y2] // voxel_length)] = 1
    return voxel_cube_cone


if __name__ == '__main__':
    """create some pairs of detections"""
    pairs = []
    df = pd.read_csv(r'C:\Users\Joe Evans\PycharmProjects\ComptonCamera\Monte Carlo\copy filepath to excel file here .xls')
    for x in df.index:
        row = df.iloc[[x]].to_numpy()[0]
        pairs.append(DetectionPair([row[1], row[2], row[3]], [row[4], row[5], row[6]], 500, 420, row[7]/2))
        print(pairs[0].lineVector)
    #pairs.append(DetectionPair([30, 15, 15], [40, 20, 20], 500, 420, np.pi/4))
    """setup the imaging area"""
    cubesize = 40
    imaging_area = np.array([cubesize, cubesize, cubesize])  # m
    voxel_length = 0.1 * 10 ** (0)  # m
    voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)
    voxel_cube = np.zeros(voxels_per_side, dtype=int)
    checks_per_side = 8
    pairs_grouped = np.array_split(pairs, (len(pairs)//25)+1)
    with Pool(multiprocessing.cpu_count()) as p:
        t = tqdm(total=len(pairs))
        for cone_group in pairs_grouped:
            args = [(imaging_area, voxel_length, voxels_per_side, checks_per_side, cone_group[i]) for i in range(len(cone_group))]
            for x in p.imap_unordered(calculate_voxel_cone_cube, iterable=args):
                t.update()
                voxel_cube += x
            args = 0
        t.close()

    #voxel_cube = np.sum(cone_list, axis=0)
    cone_list = 0
    print(np.max(voxel_cube))
    print(np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64)*voxel_length)
    #print(np.array(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape), dtype=np.float64)*voxel_length)

    # and plot everything

    #plot_3d(view_only_intersections=True)
    mayavi_plot_3d(voxel_cube, view_only_intersections=True)
