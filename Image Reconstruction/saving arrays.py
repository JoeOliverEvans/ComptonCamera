import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from mayavi import mlab


def process_2d():
    plane = voxel_cube[:, :, int(plane_z / voxel_length)]
    maxpoint = np.array(np.unravel_index(np.argmax(plane), plane.shape), dtype=np.float64) * voxel_length
    print(maxpoint)
    print(source_location)
    plt.figure(dpi=600)
    image1 = plt.imshow(plane, cmap='rainbow')
    plt.title(
        f"source location: {source_location}cm, plane location: {str(maxpoint).replace('(', '[').replace(')', ']')}cm")
    tick_locations = np.arange(0, voxel_cube.shape[0], 10)
    plt.xticks(tick_locations, np.array(tick_locations / voxel_length, dtype=int))
    plt.yticks(tick_locations, np.array(tick_locations / voxel_length, dtype=int))
    plt.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.colorbar(label="Number of cones per pixel")
    # plt.scatter(maxpoint[1], maxpoint[0], color='green')
    plt.tight_layout()
    timedate = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/', ':').replace(':', '-')
    plt.savefig(f'Plots/2d_reconstruction_save{timedate}.png')
    plt.show()


def process_3d():
    max_intersections_arguments = np.array(np.argwhere(voxel_cube == np.max(voxel_cube)))
    print(max_intersections_arguments * voxel_length)
    top10percent = np.array(np.argwhere(voxel_cube >= np.max(voxel_cube)*0.95), dtype=np.float64) * voxel_length
    print(top10percent)
    print(np.sum(top10percent, axis=0)/len(top10percent))
    print("average of top 10%")
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]

    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, voxel_cube[c, v, b], mode='cube',
                  color=(1, 0, 0), scale_mode='none', scale_factor=voxel_length)
    max_intersections_arguments = np.array(np.argwhere(voxel_cube >= np.max(voxel_cube)*0.95))
    c = max_intersections_arguments[:, 0]
    v = max_intersections_arguments[:, 1]
    b = max_intersections_arguments[:, 2]
    mlab.points3d(c * voxel_length, v * voxel_length, b * voxel_length, voxel_cube[c, v, b], mode='cube',
                  scale_mode='none', scale_factor=voxel_length, opacity=0.05, colormap='rainbow')
    mlab.axes(xlabel='x', ylabel='y', zlabel='z',
              extent=(0, np.array(voxel_cube.shape[0]/voxel_length, dtype=int), 0, np.array(voxel_cube.shape[1]/voxel_length, dtype=int), 0, np.array(voxel_cube.shape[2]/voxel_length, dtype=int)),
              nb_labels=8)
    mlab.show()


if __name__ == '__main__':
    # get file data
    file1 = r"SavedVoxelCubes\20-03-2023 14-46-56+(80, 80, 20).txt"
    file2 = r"SavedVoxelCubes\20-03-2023 14-50-52+(80, 80, 20).txt"
    loaded_arr = np.loadtxt(file1)
    loaded_arr2 = np.loadtxt(file2)
    zs = 20
    voxel_length = 1
    plane_z = 10
    # This is a 2D array - need to convert it to the original
    load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // zs, zs)

    load_original_arr2 = loaded_arr2.reshape(
        loaded_arr2.shape[0], loaded_arr2.shape[1] // zs, zs)

    # check
    print("shape of arr: ", (60, 60, 15))
    print("shape of load_original_arr: ", load_original_arr.shape)

    # check if both arrays are same or not:

    voxel_cube = load_original_arr + load_original_arr2
    source_location = np.array(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape),
                               dtype=np.float64) * voxel_length

    process_2d()
    process_3d()
