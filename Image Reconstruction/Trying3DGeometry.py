import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import matplotlib

matplotlib.use("TkAgg")

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
        self.scatterAngle = angle#CalculateScatterAngle(initial_energy, absorption_energy)


def calculate_cone_z(pair_of_detections, x, y):
    """
    :param pair_of_detections: the pair of detections that produce the cone
    :param x:
    :param y:
    :return: z1 and z2 which contain the z value, there are two as one of them will error
    """
    a = pair_of_detections.lineVector[0]
    b = pair_of_detections.lineVector[1]
    c = pair_of_detections.lineVector[2]
    t = pair_of_detections.scatterAngle
    g = np.cos(t) * (a ** 2 + b ** 2 + c ** 2) ** (1 / 2)
    d = (c ** 2 / g ** 2) - 1
    e = (2 * a * x * c + b * y * c) / (g ** 2)
    f = (((a * x) ** 2 + (b * y) ** 2 + 2 * a * x * b * y) / (g ** 2)) - x ** 2 - y ** 2
    z1 = (-e - np.sqrt(e ** 2 - 4 * d * f)) / (2 * d)
    z2 = (-e + np.sqrt(e ** 2 - 4 * d * f)) / (2 * d)
    return z1, z2


"""create some pairs of detections"""
pairs = []
firstpair = DetectionPair([49, 50, 10], [40, 50, 0], 500, 420, np.pi/4)
secondpair = DetectionPair([80, 50, 10], [80, 50, 0], 500, 300, 0.6435011088)
thirdpair = DetectionPair([50, 10, 10], [50, 10, 0], 500, 400, np.pi/4)
pairs.append(firstpair)
pairs.append(secondpair)
pairs.append(thirdpair)
print(firstpair.scatterAngle)
print(firstpair.lineVector)

"""setup the imaging area"""
cubesize = 100
imaging_area = np.array([cubesize, cubesize, cubesize])  # m
voxel_length = 2 * 10 ** (0)  # m
voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)
print(voxels_per_side)
voxel_cube = np.zeros((voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]), dtype=int)

print(firstpair.scatterPosition)

offsets = [[0, 0], [0, voxel_length], [voxel_length, 0], [voxel_length, voxel_length]]

"""populate the voxel cube with the detections of the cones"""
for pair in pairs:
    for x_index in np.arange(0, voxels_per_side[0], 1):
        for y_index in np.arange(0, voxels_per_side[1], 1):
            for z_index in np.arange(0, voxels_per_side[2], 1):
                invoxel = False
                for x_offset in np.linspace(0, voxel_length, 5):
                    for y_offset in np.linspace(0, voxel_length, 5):
                        x = x_index * voxel_length - pair.scatterPosition[0] + x_offset
                        y = y_index * voxel_length - pair.scatterPosition[1] + y_offset
                        z1, z2 = calculate_cone_z(pair, x, y)
                        z_argument1 = (z1 // voxel_length) + pair.scatterPosition[2]
                        z_argument2 = (z2 // voxel_length) + pair.scatterPosition[2]
                        if z_argument1 != z_argument2:
                            if z_index <= z_argument1 < z_index+voxel_length:   # incrementing the number of cones in voxel
                                voxel_cube[x_index, y_index, z_index] = \
                                    voxel_cube[x_index, y_index, z_index] + 1
                                invoxel = True
                            if z_index <= z_argument2 < z_index+voxel_length:
                                voxel_cube[x_index, y_index, z_index] = \
                                    voxel_cube[x_index, y_index, z_index] + 1
                                invoxel = True
                        else:
                            if z_index <= z_argument1 < z_index+voxel_length:   # incrementing the number of cones in voxel
                                voxel_cube[x_index, y_index, z_index] = \
                                    voxel_cube[x_index, y_index, z_index] + 1
                                invoxel = True
                        if invoxel:
                            break
                    if invoxel:
                        break


print(np.max(voxel_cube))
#(np.shape(voxel_cube))
print(np.unravel_index(np.argmax(voxel_cube), voxel_cube.shape))

'''source_position = np.argmax(voxel_cube) * voxel_length
print(source_position)'''


# set the colors of each object
colors = np.empty(voxel_cube.shape, dtype=object)
cones = np.where(voxel_cube < 1, voxel_cube, False)

# and plot everything
view_only_intersections = True
if view_only_intersections:
    intersections = voxel_cube > 2
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
