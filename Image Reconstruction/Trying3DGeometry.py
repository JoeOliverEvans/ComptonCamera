import math

import matplotlib;matplotlib.use("TkAgg")
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from mayavi import mlab
import warnings


warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in arcsin")

electron_mass = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  # in keV


def CalculateScatterAngle(initial_energy, final_energy):
    """
    :param final_energy: keV
    :param initial_energy: keV
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
        self.lineVector = np.array(
            (np.array(self.scatterPosition) - np.array(self.absorptionPosition)) / np.linalg.norm(
                np.array(self.scatterPosition) - np.array(self.absorptionPosition)), dtype=np.float64)
        self.absorptionEnergy = np.float64(absorption_energy)
        self.scatterAngle = np.float64(angle)  # CalculateScatterAngle(initial_energy, absorption_energy)


'''
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
    g = np.cos(t) * np.sqrt(a ** 2 + b ** 2 + c ** 2)
    d = (c ** 2 / g ** 2)
    e = (2 * a * x * c + b * y * c) / (g ** 2)
    f = (((a * x) ** 2 + (b * y) ** 2 + 2 * a * x * b * y) / (g ** 2)) - x ** 2 - y ** 2
    h = e ** 2 - 4 * d * f
    z1 = ((-e - np.sqrt(h)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    z2 = ((-e + np.sqrt(h)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    return z1, z2
'''


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
    max_intersections_arguments = np.array(np.argwhere(voxel_cube == np.max(voxel_cube)))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    # np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, voxel_cube[c, v, b], mode='cube',
                  color=(0, 1, 0), scale_mode='none', scale_factor=voxel_length)
    '''max_intersections_arguments = np.array(np.argwhere(voxel_cube >= np.max(voxel_cube_maya) - 5))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    #np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c*voxel_length,v*voxel_length,b*voxel_length, voxel_cube[c, v, b], mode='cube', scale_mode='none', scale_factor=voxel_length, opacity=0.1, colormap='autumn')'''
    mlab.axes(xlabel='x', ylabel='y', zlabel='z', extent=(0, 40, 0, 40, 0, 40), nb_labels=8)
    mlab.show()


'''
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
    return voxel_cube_cone'''


def generate_points(imaging_area, pair_of_detections, voxel_length, weight=1):
    """
    Uses Amber's cone equation to calculate points in a cone that is not pointing in the correct direction
    :param weight: Weighting for uncertainties
    :param pair_of_detections: Detection Pair object
    :return points: list containing 4 vectors, forth value is a weighting for uncertainty
    """
    def rotation_matrix(vec1, vec2):
        """
        Cara's rotation matrix code from one vector to anther vector
        :param vec1: vector of the z axis
        :param vec2: vector of the cone vector
        :return rotation_matrix: 3x3 matrix
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    vect1 = [0, 0, 1]
    vect2 = pair_of_detections.lineVector

    theta_c = pair_of_detections.scatterAngle
    edge_points = [[0, 0, 0], [imaging_area[0], 0, 0], [0, imaging_area[1], 0], [imaging_area[0], imaging_area[1], 0], [0, 0, imaging_area[2]], [imaging_area[0], 0, imaging_area[2]], [0, imaging_area[1], imaging_area[2]], [imaging_area[0], imaging_area[1], imaging_area[2]]]
    R_max = 0
    for edge_point in edge_points:
        magnitude = np.linalg.norm(edge_point-pair_of_detections.scatterPosition)
        if magnitude > R_max:
            R_max = magnitude

    #R_max = imaging_area[0]    # This corresponds to the distance of a point on the cone from the origin

    R = np.arange(0, R_max, voxel_length/2)
    points = np.array([1, 1, 1, 1])
    weight = 1
    for r in R:
        Theta_size = 2 * np.arcsin((voxel_length) / (r * np.sin(theta_c)))
        if math.isnan(Theta_size):
            Theta_size = np.pi
        Theta = np.linspace(0, 2 * np.pi, 2 * int(np.abs(2 * np.pi // Theta_size) + 1))
        for t in Theta:
            Rot = rotation_matrix(vect1, vect2)
            rotated_point = Rot.dot([r * np.sin(theta_c) * np.cos(t), r * np.sin(theta_c) * np.sin(t), np.cos(theta_c) * r])
            rotated_translated_point = rotated_point + pair_of_detections.scatterPosition
            if 0 <= rotated_translated_point[0] < imaging_area[0] and 0 <= rotated_translated_point[1] < imaging_area[1] and 0 <= rotated_translated_point[2] < \
                    imaging_area[2]:
                point = np.array([*list(rotated_translated_point), weight])
                points = np.row_stack((points, point))

    points = np.delete(points, 0, axis=0)
    return points


def calculate_cone_polars(imaging_area, pair_of_detections, voxel_length):
    """
    Calculates points for a cone in the correct orientation and starting position within the imaging area
    :param imaging_area: the size of the imaging area
    :param pair_of_detections: Detection Pair object
    :return: translated_rotated_points: list containing 4 vectors, forth value is a weighting for uncertainty
    """
    translated_rotated_points = generate_points(imaging_area, pair_of_detections, voxel_length)
    #translated_rotated_points = rotate_to_vector_and_translate(points, pair_of_detections, voxel_length, imaging_area)
    return translated_rotated_points


def calculate_voxel_cone_cube(arg):
    imaging_area, voxel_length, voxels_per_side, checks_per_side, pair_of_detections = arg[0], arg[1], arg[2], arg[3], \
                                                                                       arg[4]
    cone_points_with_weighting = calculate_cone_polars(imaging_area, pair_of_detections, voxel_length)
    voxel_cube_cone = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)
    for point in cone_points_with_weighting:
        voxel_cube_cone[int(point[0] // voxel_length),
                        int(point[1] // voxel_length),
                        int(point[2] // voxel_length)] = point[3]
    return voxel_cube_cone


if __name__ == '__main__':
    """create some pairs of detections"""
    pairs = []
    df = pd.read_csv(
        r'C:\Users\joeol\Documents\Computing year 2\ComptonCamera\Monte Carlo\copy filepath to excel file here .xls')
    for x in range(16):
        row = df.iloc[[x]].to_numpy()[0]
        pairs.append(DetectionPair([row[1], row[2], row[3]], [row[4], row[5], row[6]], 500, 420, row[7]))
    # pairs.append(DetectionPair([30, 15, 15], [40, 20, 20], 500, 420, np.pi/4))
    """setup the imaging area"""
    cubesize = 40
    imaging_area = np.array([cubesize, cubesize, cubesize])  # m
    voxel_length = 0.25 * 10 ** (0)  # m
    voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)
    voxel_cube = np.zeros(voxels_per_side, dtype=int)
    checks_per_side = 4
    pairs_grouped = np.array_split(pairs, (len(pairs) // 25) + 1)
    with Pool(multiprocessing.cpu_count()) as p:
        t = tqdm(total=len(pairs))
        for cone_group in pairs_grouped:
            args = [(imaging_area, voxel_length, voxels_per_side, checks_per_side, cone_group[i]) for i in
                    range(len(cone_group))]
            for x in p.imap_unordered(calculate_voxel_cone_cube, iterable=args):
                t.update()
                voxel_cube += x
            del args
        t.close()

    # voxel_cube = np.sum(cone_list, axis=0)
    cone_list = 0
    print(np.max(voxel_cube))
    # print(np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length)
    # print(np.array(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape), dtype=np.float64)*voxel_length)

    # and plot everything

    # plot_3d(view_only_intersections=True)
    mayavi_plot_3d(voxel_cube, view_only_intersections=True)
