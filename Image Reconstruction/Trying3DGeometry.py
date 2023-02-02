import matplotlib.pyplot as plt
import numpy as np
import scipy.constants


# TODO define class of coincident detections

electron_mass = scipy.constants.electron_mass()


class DetectionPair:
    def __init__(self, xScatter, yScatter, zScatter, xAbsorption, yAbsorption, zAdsorption, InitialEnergy, AbsorptionEnergy):
        self.ScatterPosition = [xScatter, yScatter, zScatter]
        self.AbsorptionPosition = [xAbsorption, yAbsorption, zAdsorption]
        self.LineEquation =
        self.AbsorptionEnergy = AbsorptionEnergy
        self.ScatterAngle = self.CalculateScatterAngle(InitialEnergy, AbsorptionEnergy)

    def CalculateScatterAngle(self, InitialEnergy, AbsorptionEnergy):
        """
        :param InitialEnergy: Photon energy emitted by the source
        :param AbsorptionEnergy: Photon energy at the final detector
        :return: Compton Scattering Angle
        """
        return np.arccos(1 - electron_mass * ((1/AbsorptionEnergy) - (1/InitialEnergy)))




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
