import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from mayavi import mlab
import scipy
import pandas as pd


def process_2d(matrix):
    plane = matrix[:, :, int(plane_z / voxel_length)]
    plt.figure(dpi=600)
    image1 = plt.imshow(np.transpose(plane), cmap='rainbow')
    plt.title(
        f"Actual source location: {real_source_location}, Found source location{source_location}cm")
    tick_locations = np.arange(0, matrix.shape[0], 10)
    plt.xticks(tick_locations, np.array(tick_locations / voxel_length, dtype=int))
    plt.yticks(tick_locations, np.array(tick_locations / voxel_length, dtype=int))
    plt.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.colorbar(label="Number of cones per pixel")
    # plt.scatter(maxpoint[1], maxpoint[0], color='green')
    plt.tight_layout()
    timedate = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/', ':').replace(':', '-')
    plt.savefig(f'Plots/2d_reconstruction_save{timedate}.png')
    plt.show()


def process_3d(matrix):
    max_intersections_arguments = np.array(np.argwhere(matrix == np.max(matrix)))
    cutoff = 0.001
    top10percent = np.array(np.argwhere(matrix >= np.max(matrix)*cutoff), dtype=np.float64) * voxel_length
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]

    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, matrix[c, v, b], mode='cube',
                  color=(1, 0, 0), scale_mode='none', scale_factor=voxel_length)
    max_intersections_arguments = np.array(np.argwhere(matrix >= np.max(matrix)*cutoff))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, matrix[c, v, b], mode='cube',
                  scale_mode='none', scale_factor=voxel_length, opacity=0.2, colormap='rainbow')
    mlab.axes(xlabel='x', ylabel='y', zlabel='z',
              extent=(0, np.array(matrix.shape[0]*voxel_length, dtype=int), 0, np.array(matrix.shape[1]*voxel_length, dtype=int), 0, np.array(matrix.shape[2]*voxel_length, dtype=int)),
              nb_labels=8)
    mlab.show()


def clustering(matrix, min_number_of_labels):
    """
    Returns the labeled array and a list of labels which have over the minimum number of labels
    :param min_number_of_labels:
    :param matrix:
    :return:
    """
    connection_matrix = [[[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]],
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]],
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]]
    labelled_matrix = scipy.ndimage.label(matrix, connection_matrix)
    label_list = significant_labels(labelled_matrix[0], labelled_matrix[1], min_number_of_labels)
    return np.array(labelled_matrix[0]), label_list


def significant_labels(matrix, number_of_labels, min_number_of_labels):
    relevant_labels = []
    for x in range(number_of_labels+1):
        if np.count_nonzero(matrix == x) >= min_number_of_labels and x != 0:
            relevant_labels.append(x)
    return relevant_labels


def variance(matrix):
    matrix_max = [0, 0, 0]
    matrix_min = [*matrix.shape]
    for coord in np.argwhere(matrix > 0):
        for axis, value in enumerate(coord):
            if matrix_max[axis] < value:
                matrix_max[axis] = value
            if matrix_min[axis] > value:
                matrix_min[axis] = value
    return np.array(matrix_max) - np.array(matrix_min)


if __name__ == '__main__':
    # get file data
    real_source_location = '[40, 40, 20]'
    file1 = r"SavedVoxelCubes\experimentalabsorptionscatter24thMarNewGeometryBothFiles.parquet29-03-2023 17-08-18+(80, 80, 40).txt"
    file2 = r"SavedVoxelCubes\experimentalscatterscatter24thMarNewGeometryBothFiles.parquet29-03-2023 17-15-56+(80, 80, 40).txt"
    loaded_arr = np.loadtxt(file1)
    loaded_arr2 = np.loadtxt(file2)
    zs = 40
    voxel_length = 1
    plane_z = 20
    # This is a 2D array - need to convert it to the original
    load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // zs, zs)

    load_original_arr2 = loaded_arr2.reshape(
        loaded_arr2.shape[0], loaded_arr2.shape[1] // zs, zs)

    # check
    print("shape of load_original_arr: ", load_original_arr.shape)

    # check if both arrays are same or not:

    voxel_cube = load_original_arr + load_original_arr2
    print(np.max(voxel_cube))
    voxel_cube = voxel_cube[:, :, :30]

    source_location = np.array(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape),
                               dtype=np.float64) * voxel_length
    print(np.shape(voxel_cube))
    cluster_locations, labels = clustering(np.where(voxel_cube >= np.max(voxel_cube) * 0.85, 1, 0), 5)
    print("labels" + str(labels))
    clustered_voxel_cube = np.zeros(np.shape(voxel_cube))

    source_locations = pd.DataFrame(columns=['Max Value', 'Size', 'Max Location', 'Centre of mass', 'Max Variance'])
    for label in labels:
        cluster = np.where(cluster_locations == label, voxel_cube, 0)
        max_cluster_index = np.array(np.argwhere(cluster == np.max(cluster)))
        source_locations.loc[len(source_locations)] = [np.max(cluster),
                                                       np.count_nonzero(cluster),
                                                       max_cluster_index * voxel_length,
                                                       np.round(scipy.ndimage.center_of_mass(cluster) * voxel_length, 2),
                                                       variance(cluster) * voxel_length]
        clustered_voxel_cube += cluster
    pd.options.display.max_columns = 500
    print(source_locations.sort_values(['Size'], ascending=False))

    process_2d(voxel_cube)
    process_3d(clustered_voxel_cube)
