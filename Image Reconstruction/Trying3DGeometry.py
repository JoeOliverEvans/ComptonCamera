from datetime import datetime
import math
import matplotlib
import multiprocessing
import matplotlib.pylab as plt
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


def CalculateScatterAngle(initial_energy, scatter_energy_deposited):
    """
    :param final_energy: keV
    :param initial_energy: keV
    :return: Compton Scattering Angle in radians
    """
    return np.arccos(
        1 - ((1/(initial_energy-scatter_energy_deposited)) - 1/initial_energy) * electron_mass)


class DetectionPair:
    def __init__(self, scatter_position, absorption_position, initial_energy, scatter_energy):
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
        self.scatterEnergy = np.float64(scatter_energy)
        self.scatterAngle = CalculateScatterAngle(initial_energy, scatter_energy)


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
    max_intersections_arguments = np.array(np.argwhere(voxel_cube_maya == np.max(voxel_cube_maya)))
    print(max_intersections_arguments * voxel_length)
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]

    # np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, voxel_cube[c, v, b], mode='cube',
                  color=(0, 1, 0), scale_mode='none', scale_factor=voxel_length)
    max_intersections_arguments = np.array(np.argwhere(voxel_cube_maya > 0))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    # np.array(np.argwhere(voxel_cube == np.max(voxel_cube)), dtype=np.float64) * voxel_length
    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, voxel_cube[c, v, b], mode='cube',
                  scale_mode='none', scale_factor=voxel_length, opacity=0.1, colormap='autumn')
    mlab.axes(xlabel='x', ylabel='y', zlabel='z', extent=(0, imaging_area[0], 0, imaging_area[1], 0, imaging_area[2]),
              nb_labels=8)
    mlab.show()


def rotation_matrix(vec1, vec2):
    """
    Cara's rotation matrix code from one vector to another vector
    :param vec1: vector of the z axis
    :param vec2: vector of the cone vector
    :return rotation_matrix: 3x3 matrix
    """
    if vec1 == list(vec2 / np.linalg.norm(vec2)):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def calculate_cone_polars(imaging_area, pair_of_detections, voxel_length):
    """
    Calculates points for a cone in the correct orientation and starting position within the imaging area
    :param imaging_area: the size of the imaging area
    :param pair_of_detections: Detection Pair object
    :return translated_rotated_points list containing 4 vectors, forth value is a weighting for uncertainty
    """
    vect1 = [0, 0, 1]  # z axis vector, tje cone points here by default
    vect2 = pair_of_detections.lineVector  # Vector to rotate cone axis to

    theta_c = pair_of_detections.scatterAngle

    """find the max value of R, room for great improvement"""
    '''R_max = 0
    for edge_point in edge_points:
        magnitude = np.linalg.norm(edge_point - pair_of_detections.scatterPosition)
        if magnitude > R_max:
            R_max = magnitude'''
    edge_points = [[0, 0, 0], [imaging_area[0], 0, 0], [0, imaging_area[1], 0], [imaging_area[0], imaging_area[1], 0],
                   [0, 0, imaging_area[2]], [imaging_area[0], 0, imaging_area[2]],
                   [0, imaging_area[1], imaging_area[2]], [imaging_area[0], imaging_area[1], imaging_area[2]]]
    R_max = 0
    for edge_point in edge_points:
        magnitude = np.linalg.norm(np.array(edge_point) - pair_of_detections.scatterPosition)
        if magnitude > R_max:
            R_max = magnitude
    R_min = 0
    '''if 0 <= pair_of_detections.scatterPosition[0] < imaging_area[0] and 0 <= pair_of_detections.scatterPosition[1] < \
            imaging_area[1] \
            and 0 <= pair_of_detections.scatterPosition[2] < imaging_area[2]:
        R_min = 0
    else:
        R_min = abs(pair_of_detections.scatterPosition[2] - imaging_area[2])'''
    R = np.arange(R_min, R_max, voxel_length / 2)
    points = []  # Creates arroy with the right shape
    weight = 1  # will be changed for Amber's uncertainty

    Rot = rotation_matrix(vect1, vect2)
    counter = 0
    for r in R:
        point_per_circle_in_area = False
        Theta_size = 2 * np.arcsin((voxel_length) / (r * np.sin(theta_c)))
        if math.isnan(Theta_size):
            Theta_size = np.pi
        Theta = np.linspace(0, 2 * np.pi, 2 * int(np.abs(2 * np.pi // Theta_size) + 1))
        for t in Theta:
            rotated_point = Rot.dot(
                [r * np.sin(theta_c) * np.cos(t), r * np.sin(theta_c) * np.sin(t), np.cos(theta_c) * r])
            rotated_translated_point = rotated_point + pair_of_detections.scatterPosition
            if 0 <= rotated_translated_point[0] < imaging_area[0] and 0 <= rotated_translated_point[1] < imaging_area[1] \
                    and 0 <= rotated_translated_point[2] < imaging_area[2]:
                points.append(rotated_translated_point)
                counter += 1
                point_per_circle_in_area = True
        if not point_per_circle_in_area and len(points) > 0:
            break

    points = np.reshape(points, (counter, 3))
    return points


def calculate_voxel_cone_cube(arg):
    """
    Creates a matrix like voxel_cube but only contains information for one pair of detections
    :param arg: imaging_area, voxel_length, voxels_per_side, pair_of_detections; required for multithreading
    :return voxel_cube_cone: Populated matrix for a pair of detections
    """
    imaging_area, voxel_length, voxels_per_side, pair_of_detections = arg[0], arg[1], arg[2], arg[3]
    cone_points_with_weighting = calculate_cone_polars(imaging_area, pair_of_detections, voxel_length)
    voxel_cube_cone = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)
    for point in cone_points_with_weighting:  # Fill matrix with weighting at correct points
        voxel_cube_cone[int(point[0] // voxel_length),
                        int(point[1] // voxel_length),
                        int(point[2] // voxel_length)] = 1
    return voxel_cube_cone


def save_matrix(voxelcube):
    arr_reshaped = voxelcube.reshape(voxelcube.shape[0], -1)

    # saving array
    timeanddate = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/', ':').replace(':', '-')
    np.savetxt(f"SavedVoxelCubes/{timeanddate}+{np.shape(voxelcube)}.txt", arr_reshaped)

    # get file data
    loaded_arr = np.loadtxt(f"SavedVoxelCubes/{timeanddate}+{np.shape(voxelcube)}.txt")

    # This is a 2D array - need to convert it to the original
    load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // voxelcube.shape[2], voxelcube.shape[2])

    # check
    print("shape of arr: ", voxelcube.shape)
    print("shape of load_original_arr: ", load_original_arr.shape)

    # check if both arrays are same or not:
    if (load_original_arr == voxelcube).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")
    return


if __name__ == '__main__':
    """reading in results from csv"""
    pairs = []
    df = pd.read_parquet(r'C:\Users\joeol\Documents\Computing year 2\ComptonCameraNew\Image Reconstruction\Data\experimentalscatterscatter21thMarScatterAbsorb2Source.parquet')

    print(len(df))
    print(df.head(5))
    print(df["scatter energy"].max())
    print(df["scatter energy"].min())

    z_plane = 10
    source_z = -35.4

    for x in range(len(df)):
        row = df.iloc[[x]].to_numpy()[0]
        pairs.append(
           DetectionPair(np.array(row[1]) + np.array([40, 40, z_plane-source_z]), np.array(row[3]) + np.array([40, 40, z_plane-source_z]), 662, row[0] * 1000))
    #pairs.append(DetectionPair([20, 50, 10], [30, 40, 0], 662, 100))
    #print(pairs[0].scatterPosition)
    #pairs.append(DetectionPair([24.5, 24.5, 46], [20, 6.5, 89], 662, 177))
    print(pairs[0].scatterPosition)
    print(pairs[0].absorptionPosition)
    print(pairs[0].lineVector)
    print(pairs[0].scatterAngle)
    '''pairs.append(DetectionPair([50, 10, 10], [50, 10, 0], 662, 500, np.arctan(1/1)))'''
    """setup the imaging area"""
    imaging_area = np.array([80, 80, 20])
    voxel_length = 1 * 10 ** (0)  # units matching cub_size
    voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)
    voxel_cube = np.zeros(voxels_per_side, dtype=int)

    pairs_grouped = np.array_split(pairs, (len(pairs) // 100) + 1)  # Split pairs list to save RAM, may be redundant

    """setup of the pool which allows multiple instances of python execute functions (uses all CPU cores)"""
    with Pool(multiprocessing.cpu_count()) as p:
        pbar = tqdm(total=len(pairs))  # sets up progress bar
        for cone_group in pairs_grouped:
            args = [(imaging_area, voxel_length, voxels_per_side, cone_group[i]) for i in
                    range(len(cone_group))]
            for x in p.imap_unordered(calculate_voxel_cone_cube, iterable=args):  # Distribute pairs of detections to
                # workers to compute simultaneously
                pbar.update()
                voxel_cube += x  # Add together results from workers as they arrive, if at the end numpy gets upset
            del args
        pbar.close()
    save_matrix(voxel_cube)

    # print(voxel_cube[19, 14, 57])
    # print(voxel_cube[40, 40, 40])
    cut_cube = voxel_cube[:, :, :]
    print(np.max(cut_cube))
    print(np.shape(cut_cube))
    print(np.array(np.unravel_index(np.argmax(cut_cube), cut_cube.shape), dtype=np.float64) * voxel_length)

    plane = voxel_cube[:, :, int(z_plane / voxel_length)]

    plt.figure(dpi=600)
    image1 = plt.imshow(np.transpose(plane), cmap='rainbow')
    maxpoint = np.unravel_index(np.argmax(plane), plane.shape)
    print(maxpoint)
    plt.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.colorbar()
    # plt.scatter(maxpoint[1], maxpoint[0], color='green')
    plt.tight_layout()
    timedate = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/', ':').replace(':', '-')
    plt.savefig(f'Plots/2d_reconstruction_save{timedate}.png')
    plt.show()


    # plot_3d(view_only_intersections=True)
    mayavi_plot_3d(voxel_cube[:, :, :], view_only_intersections=True)
