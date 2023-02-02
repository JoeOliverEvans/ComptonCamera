import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

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


firstpair = DetectionPair([1, 1, 1], [0, 0, 0], 500, 180)
print(firstpair.scatterAngle)
print(firstpair.lineVector)

'''
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxelarray = cube1 | cube2 | link
print(voxelarray)
# set the colors of each object
colors = np.empty(voxelarray.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

plt.show()
'''
