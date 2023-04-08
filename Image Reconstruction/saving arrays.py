import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from mayavi import mlab
import scipy
import pandas as pd


def process_2d(matrix, dataframe):
    matrix = matrix[:, ::-1, :]     # fix y direction for matplotlib
    plane_xy = np.transpose(matrix[:, :, int(plane_z / voxel_length)])     # fix x direction for matplotlib
    plane_yz = (matrix[int(np.shape(matrix)[0]/2 - 1), :, :])
    max_val = np.max([np.max(plane_xy), np.max(plane_yz)])
    min_val = np.min([np.min(plane_xy), np.min(plane_yz)])
    print(np.unravel_index(np.argmax(matrix[:, :, :25]), np.shape(matrix[:, :, :25])))

    print(plane_yz[39, 17])
    print(plane_yz[39, 16])
    print(np.unravel_index(np.argmax(plane_xy), np.shape(plane_xy)))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 4.2), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)
    fig.suptitle(f"Actual location: {real_source_location}, Found location: "
                 f"{list(source_locations['Centre of mass'].iloc[0])} cm,\n"
                 f"Variance: {list(source_locations['Max Variance'].iloc[0])} cm")
    im1 = ax[0].imshow(plane_xy, vmin=min_val, vmax=max_val, cmap='rainbow')
    im2 = ax[1].imshow(plane_yz, vmin=min_val, vmax=max_val, cmap='rainbow')
    ax[0].set_xticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])-1)
    ax[0].set_xticklabels(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80]) + graphxoffset)
    ax[0].set_yticks(np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])-1)
    ax[0].set_yticklabels(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80][::-1]) + graphyoffset)
    ax[1].set_xticks(np.array([0, 20, 40, 60, 80])-1)
    ax[1].set_xticklabels(np.round(np.array([0, 10, 20, 30, 40]) + graphzoffset, 1))
    ax[0].set_ylabel('y (cm)')
    ax[0].set_xlabel('x (cm)')
    ax[1].set_xlabel('z (cm)')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.118, 0.03, 0.753])
    plt.colorbar(im2, cax=cbar_ax, label="Number of cones per voxel")
    timedate = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/', ':').replace(':', '-')
    plt.savefig(f'Plots/2d_reconstruction_save{timedate}.png', dpi=600)
    plt.show()


    '''plt.figure(dpi=600)
    image1 = plt.imshow(plane_xy, cmap='rainbow')
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
    plt.show()'''


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
    return (np.array(matrix_max) - np.array(matrix_min)) + 1


if __name__ == '__main__':
    # get file data
    file1 = r"SavedVoxelCubes\mcabsorptionscatter15thMarNoSmear2Files.parquet07-04-2023 19-41-41+(160, 160, 80).txt"
    file2 = r"SavedVoxelCubes\mcscatterscatter15thMarNoSmear2Files.parquet07-04-2023 19-32-15+(160, 160, 80).txt"
    loaded_arr = np.loadtxt(file1)
    loaded_arr2 = np.loadtxt(file2)
    zs = 80
    voxel_length = 0.5  #cm
    plane_z = 20

    graphxoffset = -40
    graphyoffset = -40
    graphzoffset = (0)-20

    offset = np.array([graphxoffset, graphyoffset, graphzoffset])

    real_source_location = f'[0, 0, {np.round(graphzoffset+20, 1)}]'

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # This is a 2D array - need to convert it to the original
    load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // zs, zs)

    load_original_arr2 = loaded_arr2.reshape(
        loaded_arr2.shape[0], loaded_arr2.shape[1] // zs, zs)

    # check
    print("shape of load_original_arr: ", load_original_arr.shape)

    # check if both arrays are same or not:

    voxel_cube = load_original_arr + load_original_arr2

    voxel_cube = voxel_cube[:, :, :60]

    print(np.max(voxel_cube))

    source_location = np.array(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape),
                               dtype=np.float64) * voxel_length
    print(np.shape(voxel_cube))
    cluster_locations, labels = clustering(np.where(voxel_cube >= np.max(voxel_cube) * 0.65, 1, 0), 1)
    print("labels" + str(labels))
    clustered_voxel_cube = np.zeros(np.shape(voxel_cube))

    source_locations = pd.DataFrame(columns=['Max Value', 'Size', 'Max Location', 'Centre of mass', 'Max Variance'])
    for label in labels:
        cluster = np.where(cluster_locations == label, voxel_cube, 0)
        max_cluster_index = np.array(np.argwhere(cluster == np.max(cluster)))
        source_locations.loc[len(source_locations)] = [np.max(cluster),
                                                       np.count_nonzero(cluster),
                                                       max_cluster_index * voxel_length + offset,
                                                       np.round(np.array(scipy.ndimage.center_of_mass(cluster)) * voxel_length + offset, 1),
                                                       variance(cluster) * voxel_length]
        clustered_voxel_cube += cluster
    pd.options.display.max_columns = 500
    source_locations = source_locations.sort_values(['Size'], ascending=False)
    print(source_locations)
    process_2d(voxel_cube, source_locations)
    process_3d(clustered_voxel_cube)
