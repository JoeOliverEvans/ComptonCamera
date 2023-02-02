import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants


electron_mass = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  # in keV

def CalculateScatterAngle(InitialEnergy, AbsorptionEnergy):
    """
    :param InitialEnergy: Photon energy emitted by the source
    :param AbsorptionEnergy: Photon energy at the final detector
    :return: Compton Scattering Angle in degrees
    """
    return np.arccos(
        1 - (electron_mass * ((InitialEnergy - AbsorptionEnergy) / (InitialEnergy * AbsorptionEnergy)))) * 180 / np.pi


class DetectionPair:
    def __init__(self, xScatter, yScatter, zScatter, xAbsorption, yAbsorption, zAdsorption, InitialEnergy,
                 AbsorptionEnergy):
        """
        :param xScatter: Coordinates of scatter
        :param yScatter:
        :param zScatter:
        :param xAbsorption: Coordinates of absorption
        :param yAbsorption:
        :param zAdsorption:
        :param InitialEnergy:
        :param AbsorptionEnergy:
        """
        self.ScatterPosition = [xScatter, yScatter, zScatter]
        self.AbsorptionPosition = [xAbsorption, yAbsorption, zAdsorption]
        self.LineVector = np.array(self.ScatterPosition) - np.array(self.AbsorptionPosition)
        self.AbsorptionEnergy = AbsorptionEnergy
        self.ScatterAngle = CalculateScatterAngle(InitialEnergy, AbsorptionEnergy)


firstpair = DetectionPair(1, 1, 1, 0, 0, 0, 500, 180)
print(firstpair.ScatterAngle)
print(firstpair.LineVector)



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
