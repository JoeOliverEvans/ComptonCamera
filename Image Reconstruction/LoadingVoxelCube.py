import numpy as np

if __name__ == '__main__':
	filename = "SavedVoxelCubes/14-03-2023 17-10-06+(20, 20, 5).txt"
	loaded_arr = np.loadtxt(filename)

	array_shape = eval('('+filename.split('(')[1].split(')')[0]+')')

	# This is a 2D array - need to convert it to the original
	load_original_arr = loaded_arr.reshape(
		loaded_arr.shape[0], loaded_arr.shape[1] // array_shape[2], array_shape[2])


