import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import matplotlib
import time
import datetime
from multiprocessing import Pool
from tqdm import tqdm

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
        self.scatterPosition = scatter_position
        self.absorptionPosition = absorption_position
        self.lineVector = (np.array(self.scatterPosition) - np.array(self.absorptionPosition)) / np.linalg.norm(
            np.array(self.scatterPosition) - np.array(self.absorptionPosition))
        self.absorptionEnergy = absorption_energy
        self.scatterAngle = angle  # CalculateScatterAngle(initial_energy, absorption_energy)


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
    z1 = ((-e - np.sqrt(e ** 2 - 4 * d * f)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    z2 = ((-e + np.sqrt(e ** 2 - 4 * d * f)) / (2 * d)) + pair_of_detections.scatterPosition[2]
    return z1, z2


def plot_3d(view_only_intersections=False):
    """
    Function to view a 3D plot of cones, can toggle to see only intersections
    :param view_only_intersections: True or False
    :return:
    """
    matplotlib.use("TkAgg")
    if view_only_intersections:
        intersections = voxel_cube > 1
        # intersections = np.where(voxel_cube > 1, voxel_cube, True)
        intersections = np.array(intersections, dtype=bool)
        colors[intersections] = 'green'

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(intersections, facecolors=colors, edgecolor='k')

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


def calculate_voxel_cone_cube(arg):
    imaging_area, voxel_length, voxels_per_side, checks_per_side, pair_of_detections = arg[0], arg[1], arg[2], arg[3], arg[4]
    x_values = np.tile(np.arange(0, imaging_area[0], voxel_length / checks_per_side) - pair_of_detections.scatterPosition[0],
                       (voxels_per_side[1] * checks_per_side, 1))
    y_values = np.tile(
        np.array(
            [np.arange(0, imaging_area[1], voxel_length / checks_per_side) - pair_of_detections.scatterPosition[1]]).transpose(),
        (1, voxels_per_side[0] * checks_per_side))
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
    firstpair = DetectionPair([40, 50, 10], [40, 50, 0], 500, 420, 0.463647609)
    secondpair = DetectionPair([80, 50, 10], [80, 50, 0], 500, 300, 0.6435011088)
    thirdpair = DetectionPair([50, 10, 10], [50, 10, 0], 500, 400, np.pi / 4)
    for n in range(0, 100):
        pairs.append(firstpair)
        pairs.append(secondpair)
        pairs.append(thirdpair)
    print(firstpair.scatterAngle)
    print(firstpair.lineVector)

    """setup the imaging area"""
    cubesize = 100
    imaging_area = np.array([cubesize, cubesize, cubesize])  # m
    voxel_length = 1 * 10 ** (0)  # m
    voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)

    #voxel_cube = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)

    points_per_voxel_side = 2
    checks_per_side = 4
    args = [(imaging_area, voxel_length, voxels_per_side, checks_per_side, pairs[i]) for i in range(len(pairs))]
    print('start')
    with Pool(multiprocessing.cpu_count()) as p:
        cone_list = list(tqdm(p.imap_unordered(calculate_voxel_cone_cube, iterable=args), total=len(pairs)))
        '''        while cone_list._number_left > 0:
            number_of_tasks_complete = len(pairs) - cone_list._number_left * cone_list._chunksize
            pbar.update(number_of_tasks_complete)
        pbar.close()'''
    '''
    for n, pair in enumerate(pairs):
        # print(calculate_voxel_cone_cube(pair))
        cone_list.append(calculate_voxel_cone_cube(pair))
        if (time.time() - t_0) > 5:
            if first:
                first = False
                time_to_completion = round((time.time() - t_0) * len(pairs) / n)
                print('Estimated time to completion : ' + str(datetime.timedelta(seconds=time_to_completion)))
            print(str(n) + ' / ' + str(len(pairs)))
            t_0 = time.time()'''
    '''     x_values = np.tile(np.arange(0, imaging_area[0], voxel_length / checks_per_side) - pair.scatterPosition[0],
                           (voxels_per_side[1] * checks_per_side, 1))
        y_values = np.tile(
            np.array(
                [np.arange(0, imaging_area[1], voxel_length / checks_per_side) - pair.scatterPosition[1]]).transpose(),
            (1, voxels_per_side[0] * checks_per_side))

        z1, z2 = calculate_cone_z(pair, x_values, y_values)
        z1_arg = -1
        z2_arg = -1
        voxel_cube_cone = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)
        for (x1, y1) in np.argwhere((0 <= z1) & (z1 < imaging_area[2])):
            counter += 1
            voxel_cube_cone[x1 // checks_per_side, y1 // checks_per_side, int(z1[x1, y1] // voxel_length)] = 1
        for (x2, y2) in np.argwhere((0 <= z2) & (z2 < imaging_area[2])):
            counter += 1
            voxel_cube_cone[x2 // checks_per_side, y2 // checks_per_side, int(z2[x2, y2] // voxel_length)] = 1'''
    '''
        for (x, y), value in np.ndenumerate(z1):
            counter+=1
            x = voxels_per_side[0] - (x+1)
            if 0 <= z1[y, x] < imaging_area[2]:
                z1_arg = int(z1[y, x] // voxel_length)
                voxel_cube_cone[x//checks_per_side, y//checks_per_side, z1_arg] = 1
            if 0 <= z2[y, x] < imaging_area[2]:
                z2_arg = int(z2[y, x] // voxel_length)
                if z1_arg != z2_arg:
                    voxel_cube_cone[x//checks_per_side, y//checks_per_side, z2_arg] = 1'''
    voxel_cube = np.sum(cone_list, axis=0)
    print(np.max(voxel_cube))

    print(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape))

    # set the colors of each object
    colors = np.empty(voxel_cube.shape, dtype=object)
    cones = np.where(voxel_cube < 1, voxel_cube, False)

    # and plot everything

    #plot_3d()
