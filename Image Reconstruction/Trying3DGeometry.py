import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import matplotlib; matplotlib.use("TkAgg")

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
    def __init__(self, scatter_position, absorption_position, initial_energy, absorption_energy):
        """
        :param scatter_position: Coordinates of scatter
        :param absorption_position: Coordinates of absorption
        :param initial_energy:
        :param absorption_energy:
        """
        self.scatterPosition = scatter_position
        self.absorptionPosition = absorption_position
        self.lineVector = (np.array(self.scatterPosition) - np.array(self.absorptionPosition))/np.linalg.norm(np.array(self.scatterPosition) - np.array(self.absorptionPosition))
        self.absorptionEnergy = absorption_energy
        self.scatterAngle = CalculateScatterAngle(initial_energy, absorption_energy)


pairs = []
firstpair = DetectionPair([26, 26, 10], [21, 21, 0], 500, 420)
secondpair = DetectionPair([71, 71, 1], [80, 80, 0], 500, 300)
thirdpair = DetectionPair([10, 71, 1], [10, 80, 0], 500, 200)
pairs.append(firstpair)
pairs.append(secondpair)
pairs.append(thirdpair)
print(firstpair.scatterAngle)
print(firstpair.lineVector)

imaging_area = 100  # m
voxel_length = 1 * 10 ** (0)  # m
voxels_per_side = int(imaging_area / voxel_length)
print(voxels_per_side)
voxel_cube = np.zeros((voxels_per_side, voxels_per_side, voxels_per_side), dtype=int)


for pair in pairs:
    for x in np.arange(-pair.scatterPosition[0], imaging_area-pair.scatterPosition[0], voxel_length):
        for y in np.arange(-pair.scatterPosition[1], imaging_area-pair.scatterPosition[1], voxel_length):
            beta = np.arctan(pair.lineVector[0]/pair.lineVector[2])
            gamma = np.arctan(pair.lineVector[1]/pair.lineVector[2])
            a = pair.lineVector[0]
            b = pair.lineVector[1]
            c = pair.lineVector[2]
            t = pair.scatterAngle
            g = np.cos(t) * (a**2 + b**2 + c**2) ** (1/2)
            d = (c**2/g**2)-1
            e = (2*a*x*c+b*y*c)/(g**2)
            f = (((a*x)**2 + (b*y)**2 + 2 * a * x * b * y)/(g**2)) - x**2 - y**2
            z1 = (-e - np.sqrt(e**2 - 4 * d * f))/(2*d)
            z2 = (-e + np.sqrt(e**2 - 4 * d * f))/(2*d)
            #z = (-np.sqrt(c**2 *(-2* a *x - 2* b *y)**2 - 4 *(a**2 *np.cos(t)**2 + b**2 *np.cos(t)**2 + c**2* np.cos(t)**2 - c**2)* (a**2* x**2* np.cos(t)**2 + a**2* y**2* np.cos(t)**2 - a**2* x**2 - 2* a *b *x *y + b**2 *x**2* np.cos(t)**2 + b**2 *y**2 *np.cos(t)**2 - b**2 *y**2 + c**2 *x**2 *np.cos(t)**2 + c**2 *y**2 *np.cos(t)**2)) - c* (-2* a *x - 2 *b *y))/(2* (a**2* np.cos(t)**2 + b**2 *np.cos(t)**2 + c**2* np.cos(t)**2 - c**2))
            #z = (np.sqrt(x**2 + y**2) / (np.tan(pair.scatterAngle)))
            z_argument1 = (z1 // voxel_length) + pair.scatterPosition[2]
            z_argument2 = (z2 // voxel_length) + pair.scatterPosition[2]
            if 0 <= z_argument1 < voxels_per_side:
                voxel_cube[(x + pair.scatterPosition[0], y + pair.scatterPosition[1], int(z_argument1))] = voxel_cube[(x + pair.scatterPosition[0], y + pair.scatterPosition[1], int(z_argument1))] + 1
            if 0 <= z_argument2 < voxels_per_side:
                voxel_cube[(x + pair.scatterPosition[0], y + pair.scatterPosition[1], int(z_argument2))] = voxel_cube[(x + pair.scatterPosition[0], y + pair.scatterPosition[1], int(z_argument2))] + 1


print(voxel_cube[21, 25, 5])
'''for pair in pairs:
    x = np.arange(0, imaging_area, voxel_length)
    y = np.arange(0, imaging_area, voxel_length)'''


x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between



# combine the objects into a single boolean arrayanotheranoth

print(np.max(voxel_cube))

# set the colors of each object
colors = np.empty(voxel_cube.shape, dtype=object)
cones = np.where(voxel_cube < 1, voxel_cube, False)

# and plot everything
view_only_intersections = True
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
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.voxels(voxel_cube, facecolors=colors, edgecolor='k')
    ax.plot([firstpair.scatterPosition[0], firstpair.scatterPosition[0]+40*firstpair.lineVector[0]],[firstpair.scatterPosition[1], firstpair.scatterPosition[1]+40*firstpair.lineVector[1]], [firstpair.scatterPosition[2], firstpair.scatterPosition[2]+40*firstpair.lineVector[2]])
    plt.show()

