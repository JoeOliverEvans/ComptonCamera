import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
#import matplotlib; matplotlib.use("TkAgg")

electron_mass = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  # in keV


def CalculateScatterAngle(initial_energy, final_energy):
    """
    :param final_energy:
    :param initial_energy:
    :return: Compton Scattering Angle in degrees
    """
    return np.arccos(
        1 - (electron_mass * ((initial_energy - final_energy) / (initial_energy * final_energy)))) * 180 / np.pi


class DetectionPair:
    def __init__(self, scatter_position, absorption_position, initial_energy, absorption_energy):
        """
        :param scatter_position: Coordinates of scatter
        :param absorption_position: Coordinates of absorption
        :param initial_energy:
        :param absorption_energy:
        """
        self.scatterPosition = scatter_position
        self.absorptionPosition = absorption_position
        self.lineVector = np.array(self.scatterPosition) - np.array(self.absorptionPosition)
        self.absorptionEnergy = absorption_energy
        self.scatterAngle = CalculateScatterAngle(initial_energy, absorption_energy)


pairs = []
firstpair = DetectionPair([31, 31, 1], [30, 30, 0], 500, 400)
secondpair = DetectionPair([71, 71, 1], [80, 80, 0], 500, 300)
#thirdpair = DetectionPair([10, 71, 1], [10, 80, 0], 500, 200)
pairs.append(firstpair)
pairs.append(secondpair)
#pairs.append(thirdpair)
print(firstpair.scatterAngle)
print(firstpair.lineVector)

imaging_area = 100  # m
voxel_length = 1 * 10 ** (0)  # m
voxels_per_side = int(imaging_area / voxel_length)
print(voxels_per_side)
voxel_cube = np.zeros((voxels_per_side, voxels_per_side, voxels_per_side), dtype=int)

for pair in pairs:
    for x in np.arange(0, imaging_area, voxel_length):
        for y in np.arange(0, imaging_area, voxel_length):
            z = pair.scatterPosition[2] + np.sqrt(
                ((x - pair.scatterPosition[0]) ** 2 + (y - pair.scatterPosition[1]) ** 2)
                * (np.tan(pair.scatterAngle)) ** 2)
            z_argument = z // voxel_length
            if z_argument < voxels_per_side:
                voxel_cube[(x, y, int(z_argument))] = voxel_cube[(x, y, int(z_argument))] + 1


'''for pair in pairs:
    x = np.arange(0, imaging_area, voxel_length)
    y = np.arange(0, imaging_area, voxel_length)'''


x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between



# combine the objects into a single boolean array

print(np.max(voxel_cube))

# set the colors of each object
colors = np.empty(voxel_cube.shape, dtype=object)
cones = np.where(voxel_cube < 1, voxel_cube, False)

# and plot everything
view_only_intersections = False
if view_only_intersections == True:
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
    ax.voxels(voxel_cube, facecolors=colors, edgecolor='k')

    plt.show()

