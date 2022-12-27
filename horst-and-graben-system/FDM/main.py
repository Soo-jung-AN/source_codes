
from msilib.schema import Condition
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import numpy.linalg as lina
import time
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
########################################################################################
L1 = 0; L2 = 35e3
H1 = 0; H2 = 5500 # 3 km spared space on the top for padding
nx = 250; ny = 40  # Selcting Grid resolution
X_lin = np.linspace(L1,L2,nx)
Y_lin = np.linspace(H1,H2,ny)
X,Y = np.meshgrid(X_lin,Y_lin) 
############################################################################################################################################################
init, extended_1km, extended_2km, extended_3km = 5814, 15930, 26049, 36249
undeform = init
deform = extended_1km
undeformed_cood = np.loadtxt("data/extension_cood0_{}.txt".format(undeform))
deformed_cood = np.loadtxt("data/extension_cood0_{}.txt".format(deform))
spin = np.loadtxt("data/angular_vel0_{}.txt".format(25188))
rad = np.loadtxt("data/radius.txt")
############################################################################################################################################################
p_num = len(undeformed_cood)
disp = deformed_cood - undeformed_cood
disp_x_list = disp[:,0]; disp_y_list = disp[:,1]
deformed_x_list = deformed_cood[:, 0]; deformed_y_list = deformed_cood[:, 1]
undeformed_x_list = undeformed_cood[:,0]; undeformed_y_list = undeformed_cood[:,1]
plt.figure()
plt.title("Grided mesh")
plt.scatter(undeformed_x_list,undeformed_y_list,c='k')
plt.scatter(X,Y,c='magenta')
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(17, 9)
##################################### Inversed distance weighted interpolation from grid nodes to particles ##########################################
start = time.time()
dx = X_lin[2]-X_lin[1]
dy = Y_lin[2]-Y_lin[1]
print('dx:',dx,'dy:',dy)
print('#particle:',len(undeformed_x_list),'mesh:',nx)
x_disp_n = 1e20 * np.ones_like(X); y_disp_n = 1e20 * np.ones_like(Y); ID_list = 1e20 * np.ones_like(X)
p = 3 # exponential order
for i in range(ny): #### particle --> node #####
    for j in range(nx):
        x_node = X[i, j]
        y_node = Y[i, j]
        dist = np.sqrt((undeformed_x_list - x_node) ** 2 + (undeformed_y_list - y_node) ** 2)
        min_dist = np.min(dist)
        min_dist_p_id = np.argmin(dist)
        ############# Nearest neighbor searching ############### (Nearest Neighbor Searching method can be selected)
        # if dist[min_dist_p_id] > 80*5:#np.max(rad)*5:
        #      continue
        # else:
        #      x_disp_n[i, j] = disp_x_list[min_dist_p_id]
        #      y_disp_n[i, j] = disp_y_list[min_dist_p_id]
        ############# IDW(inversed distance weight) interpolation ##############
        base_u = 0; base_v = 0; base_weight = 0
        for k in range(len(dist)):
            if dist[k] <= dx * 2: # Interpolation length: Radial range from a k th grid node
                weight = 1 / dist[k] ** p 
                base_weight += weight
                u = disp_x_list[k]
                base_u += u * weight
                v = disp_y_list[k]
                base_v += v * weight
                x_disp_n[i, j] = base_u / base_weight
                y_disp_n[i, j] = base_v / base_weight
ID_list[ID_list == 1e20] = np.nan
x_disp_n[x_disp_n == 1e20] = np.nan
y_disp_n[y_disp_n == 1e20] = np.nan
##################################### Finite difference method for displacement differentiation ######################################
u11 = 1e20 * np.ones_like(X); u22 = 1e20 * np.ones_like(X); u12 = 1e20 * np.ones_like(X); u21 = 1e20 * np.ones_like(X)
f11 = 1e20 * np.ones_like(X); f22 = 1e20 * np.ones_like(X); f12 = 1e20 * np.ones_like(X); f21 = 1e20 * np.ones_like(X)
E11 = 1e20 * np.ones_like(X); E21 = 1e20 * np.ones_like(X); E12 = 1e20 * np.ones_like(X); E22 = 1e20 * np.ones_like(X)
I_E = 1e20 * np.ones_like(X); II_E = 1e20 * np.ones_like(X)
x_disp_n = np.nan_to_num(x_disp_n, copy=False)
y_disp_n = np.nan_to_num(y_disp_n, copy=False)
for i in range(nx-2): # coloum
    for j in range(ny-2): # row
        j_central = j + 1
        i_central = i + 1
        u11[j_central, i_central] = (x_disp_n[j, i + 2] - x_disp_n[j, i]) / (2 * dx)
        u22[j_central, i_central] = (y_disp_n[j + 2, i] - y_disp_n[j, i]) / (2 * dy)  ####### au/aX ######
        u12[j_central, i_central] = (x_disp_n[j + 2, i] - x_disp_n[j, i]) / (2 * dy)
        u21[j_central, i_central] = (y_disp_n[j, i + 2] - y_disp_n[j, i]) / (2 * dx)
        # deformatioin gradient tensor
        f11[j_central, i_central] = u11[j_central, i_central] + 1
        f12[j_central, i_central] = u12[j_central, i_central]  ####### F = (ax/aX) ######
        f21[j_central, i_central] = u21[j_central, i_central]
        f22[j_central, i_central] = u22[j_central, i_central] + 1
        # (lagrangian) finite strain tensor
        E11[j_central, i_central] = u11[j_central, i_central] + 0.5 * (u11[j_central, i_central] ** 2 + u21[j_central, i_central] ** 2)
        E12[j_central, i_central] = 0.5 * (u12[j_central, i_central] + u21[j_central, i_central]) + 0.5 * (u11[j_central, i_central] * u12[j_central, i_central] + u21[j_central, i_central] * u22[j_central, i_central])  ####### E = 0.5(aUk/aXL + aUL/aXk + aUM/aXk*aUM/aXL) ######
        E21[j_central, i_central] = 0.5 * (u21[j_central, i_central] + u12[j_central, i_central]) + 0.5 * (u12[j_central, i_central] * u11[j_central, i_central] + u22[j_central, i_central] * u21[j_central, i_central])
        E22[j_central, i_central] = u22[j_central, i_central] + 0.5 * (u12[j_central, i_central] ** 2 + u22[j_central, i_central] ** 2)
        # invariants I
        I_E[j_central, i_central] = (f11[j_central, i_central] * f22[j_central, i_central] - f12[j_central, i_central] * f21[j_central, i_central])
        # invariants II
        II_E[j_central, i_central] = (E11[j_central, i_central] - (E11[j_central, i_central] + E22[j_central, i_central])/2) * (E22[j_central, i_central] - (E11[j_central, i_central] + E22[j_central, i_central])/2) - E21[j_central, i_central]**2#dev_e11 * dev_e22 - (dev_e12 ** 2)

for j in range(ny-2): # right
    j_right = j + 1
    u11[j_right, 0] = (x_disp_n[j_right, 1] - x_disp_n[j_right, 0]) / (dx)
    u22[j_right, 0] = (y_disp_n[j + 2, 0] - y_disp_n[j, 0]) / (2 * dy)
    u12[j_right, 0] = (x_disp_n[j + 2, 0] - x_disp_n[j, 0]) / (2 * dy)
    u21[j_right, 0] = (y_disp_n[j_right, 1] - y_disp_n[j_right, 0]) / (dx)
    u11[0, 0] = (x_disp_n[0, 1] - x_disp_n[0, 0]) / (dx) # right bottom
    u22[0, 0] = (y_disp_n[1, 0] - y_disp_n[0, 0]) / (dy)
    u12[0, 0] = (x_disp_n[1, 0] - x_disp_n[0, 0]) / (dy)
    u21[0, 0] = (y_disp_n[0, 1] - y_disp_n[0, 0]) / (dx)
    u11[-1, 0] = (x_disp_n[-1, 1] - x_disp_n[-1, 0]) / (dx) #right top
    u22[-1, 0] = (y_disp_n[-1, 0] - y_disp_n[-2, 0]) / (dy)
    u12[-1, 0] = (x_disp_n[-1, 0] - x_disp_n[-2, 0]) / (dy)
    u21[-1, 0] = (y_disp_n[-1, 1] - y_disp_n[-1, 0]) / (dx)
    # deformatioin gradient tensor
    f11[j_right, 0] = u11[j_right, 0] + 1
    f12[j_right, 0] = u12[j_right, 0]  ####### F = (ax/aX) ######
    f21[j_right, 0] = u21[j_right, 0]
    f22[j_right, 0] = u22[j_right, 0] + 1
    f11[0, 0] = u11[0, 0] + 1; f12[0, 0] = u12[0, 0] ; f21[0, 0] = u21[0, 0]; f22[0, 0] = u22[0, 0] + 1
    f11[-1, 0] = u11[-1, 0] + 1; f12[-1, 0] = u12[-1, 0] ; f21[-1, 0] = u21[-1, 0]; f22[-1, 0] = u22[-1, 0] + 1
    # (lagrangian) finite strain tensor
    E11[j_right, 0] = u11[j_right, 0] + 0.5 * (u11[j_right, 0] ** 2 + u21[j_right, 0] ** 2)
    E12[j_right, 0] = 0.5 * (u12[j_right, 0] + u21[j_right, 0]) + 0.5 * (u11[j_right, 0] * u12[j_right, 0] + u21[j_right, 0] * u22[j_right, 0])  ####### E = 0.5(aUk/aXL + aUL/aXk + aUM/aXk*aUM/aXL) ######
    E21[j_right, 0] = 0.5 * (u21[j_right, 0] + u12[j_right, 0]) + 0.5 * (u12[j_right, 0] * u11[j_right, 0] + u22[j_right, 0] * u21[j_right, 0])
    E22[j_right, 0] = u22[j_right, 0] + 0.5 * (u12[j_right, 0] ** 2 + u22[j_right, 0] ** 2)
    E11[0, 0] = u11[0, 0] + 0.5 * (u11[0, 0] ** 2 + u21[0, 0] ** 2)
    E12[0, 0] = 0.5 * (u12[0, 0] + u21[0, 0]) + 0.5 * (u11[0, 0] * u12[0, 0] + u21[0, 0] * u22[0, 0])  ####### E = 0.5(aUk/aXL + aUL/aXk + aUM/aXk*aUM/aXL) ######
    E21[0, 0] = 0.5 * (u21[0, 0] + u12[0, 0]) + 0.5 * (u12[0, 0] * u11[0, 0] + u22[0, 0] * u21[0, 0])
    E22[0, 0] = u22[0, 0] + 0.5 * (u12[0, 0] ** 2 + u22[0, 0] ** 2)
    E11[-1, 0] = u11[-1, 0] + 0.5 * (u11[-1, 0] ** 2 + u21[-1, 0] ** 2)
    E12[-1, 0] = 0.5 * (u12[-1, 0] + u21[-1, 0]) + 0.5 * (u11[-1, 0] * u12[-1, 0] + u21[-1, 0] * u22[-1, 0])  ####### E = 0.5(aUk/aXL + aUL/aXk + aUM/aXk*aUM/aXL) ######
    E21[-1, 0] = 0.5 * (u21[-1, 0] + u12[-1, 0]) + 0.5 * (u12[-1, 0] * u11[-1, 0] + u22[-1, 0] * u21[-1, 0])
    E22[-1, 0] = u22[-1, 0] + 0.5 * (u12[-1, 0] ** 2 + u22[-1, 0] ** 2)
    # invariants I
    I_E[j_right, 0] = (f11[j_right, 0] * f22[j_right, 0] - f12[j_right, 0] * f21[j_right, 0])
    I_E[0, 0] = (f11[0, 0] * f22[0, 0] - f12[0, 0] * f21[0, 0])
    I_E[-1, 0] = (f11[-1, 0] * f22[-1, 0] - f12[-1, 0] * f21[-1, 0])
    # invariants II
    II_E[j_right, 0] = (E11[j_right, 0] - (E11[j_right, 0] + E22[j_right, 0])/2) * (E22[j_right, 0] - (E11[j_right, 0] + E22[j_right, 0])/2) - E21[j_right, 0]**2#dev_e11 * dev_e22 - (dev_e12 ** 2)
    II_E[0, 0] = (E11[0, 0] - (E11[0, 0] + E22[0, 0])/2) * (E22[0, 0] - (E11[0, 0] + E22[0, 0])/2) - E21[0, 0]**2#dev_e11 * dev_e22 - (dev_e12 ** 2)
    II_E[-1, 0] = (E11[-1, 0] - (E11[-1, 0] + E22[-1, 0])/2) * (E22[-1, 0] - (E11[-1, 0] + E22[-1, 0])/2) - E21[-1, 0]**2#dev_e11 * dev_e22 - (dev_e12 ** 2)

for j in range(ny - 2):  # left
    j_left  = j + 1
    u11[j_left, -1] = (x_disp_n[j_left, -1] - x_disp_n[j_left, -2]) / (dx)
    u22[j_left, -1] = (y_disp_n[j + 2, -1] - y_disp_n[j, -1]) / (2 * dy)
    u12[j_left, -1] = (x_disp_n[j + 2, -1] - x_disp_n[j, -1]) / (2 * dy)
    u21[j_left, -1] = (y_disp_n[j_left, -1] - y_disp_n[j_left, -2]) / (dx)
    u11[0, -1] = (x_disp_n[0, -1] - x_disp_n[0, -2]) / (dx)  # LEFT bottom corner
    u22[0, -1] = (y_disp_n[1, -1] - y_disp_n[0, -1]) / (dy)
    u12[0, -1] = (x_disp_n[1, -1] - x_disp_n[0, -1]) / (dy)
    u21[0, -1] = (y_disp_n[0, -1] - y_disp_n[0, -2]) / (dx)
    u11[-1, -1] = (x_disp_n[-1, -1] - x_disp_n[-1, -2]) / (dx)  # LEFT top
    u22[-1, -1] = (y_disp_n[-1, -1] - y_disp_n[-2, -1]) / (dy)
    u12[-1, -1] = (x_disp_n[-1, -1] - x_disp_n[-2, -1]) / (dy)
    u21[-1, -1] = (y_disp_n[-1, -1] - y_disp_n[-1, -2]) / (dx)
    # deformatioin gradient tensor
    f11[j_left, -1] = u11[j_left, -1] + 1
    f12[j_left, -1] = u12[j_left, -1]  ####### F = (ax/aX) ######
    f21[j_left, -1] = u21[j_left, -1]
    f22[j_left, -1] = u22[j_left, -1] + 1
    f11[0, -1] = u11[0, -1] + 1; f12[0, -1] = u12[0, -1]; f21[0, -1] = u21[0, -1]; f22[0, -1] = u22[0, -1] + 1
    f11[-1, -1] = u11[-1, -1] + 1; f12[-1, -1] = u12[-1, -1]; f21[-1, -1] = u21[-1, -1]; f22[-1, -1] = u22[-1, -1] + 1
    # (lagrangian) finite strain tensor (Green strain)
    E11[j_left, -1] = u11[j_left, -1] + 0.5 * (u11[j_left, -1] ** 2 + u21[j_left, -1] ** 2)
    E12[j_left, -1] = 0.5 * (u12[j_left, -1] + u21[j_left, -1]) + 0.5 * (u11[j_left, -1] * u12[j_left, -1] + u21[j_left, -1] * u22[j_left, -1])
    E21[j_left, -1] = 0.5 * (u21[j_left, -1] + u12[j_left, -1]) + 0.5 * (u12[j_left, -1] * u11[j_left, -1] + u22[j_left, -1] * u21[j_left, -1])
    E22[j_left, -1] = u22[j_left, -1] + 0.5 * (u12[j_left, -1] ** 2 + u22[j_left, -1] ** 2)
    E11[0, -1] = u11[0, -1] + 0.5 * (u11[0, -1] ** 2 + u21[0, -1] ** 2)
    E12[0, -1] = 0.5 * (u12[0, -1] + u21[0, -1]) + 0.5 * (u11[0, -1] * u12[0, -1] + u21[0, -1] * u22[0, -1])
    E21[0, -1] = 0.5 * (u21[0, -1] + u12[0, -1]) + 0.5 * (u12[0, -1] * u11[0, -1] + u22[0, -1] * u21[0, -1])
    E22[0, -1] = u22[0, -1] + 0.5 * (u12[0, -1] ** 2 + u22[0, -1] ** 2)
    E11[-1, -1] = u11[-1, -1] + 0.5 * (u11[-1, -1] ** 2 + u21[-1, -1] ** 2)
    E12[-1, -1] = 0.5 * (u12[-1, -1] + u21[-1, -1]) + 0.5 * (u11[-1, -1] * u12[-1, -1] + u21[-1, -1] * u22[-1, -1])
    E21[-1, -1] = 0.5 * (u21[-1, -1] + u12[-1, -1]) + 0.5 * (u12[-1, -1] * u11[-1, -1] + u22[-1, -1] * u21[-1, -1])
    E22[-1, -1] = u22[-1, -1] + 0.5 * (u12[-1, -1] ** 2 + u22[1, -1] ** 2)
    # invariants I
    I_E[j_left, -1] = (f11[j_left, -1] * f22[j_left, -1] - f12[j_left, -1] * f21[j_left, -1])
    I_E[0, -1] = (f11[0, -1] * f22[0, -1] - f12[0, -1] * f21[0, -1])
    I_E[-1, -1] = (f11[-1, -1] * f22[-1, -1] - f12[-1, -1] * f21[-1, -1])
    # invariants II
    II_E[j_left, -1] = (E11[j_left, -1] - (E11[j_left, -1] + E22[j_left, -1]) / 2) * (E22[j_left, -1] - (E11[j_left, -1] + E22[j_left, -1]) / 2) - E21[j_left, -1] ** 2
    II_E[0, -1] = (E11[0, -1] - (E11[0, -1] + E22[0, -1]) / 2) * (E22[0, -1] - (E11[0, -1] + E22[0, -1]) / 2) - E21[0, -1] ** 2
    II_E[-1, -1] = (E11[-1, -1] - (E11[-1, -1] + E22[-1, -1]) / 2) * (E22[-1, -1] - (E11[-1, -1] + E22[-1, -1]) / 2) - E21[-1, -1] ** 2

for i in range(nx-2):
    i_top = i + 1
    u11[0, i_top] = (x_disp_n[0, i+2] - x_disp_n[0, i]) / (2*dx)
    u22[0, i_top] = (y_disp_n[1, i_top] - y_disp_n[0, i_top]) / (dy)
    u12[0, i_top] = (x_disp_n[1, i_top] - x_disp_n[0, i_top]) / (dy)
    u21[0, i_top] = (y_disp_n[0, i+2] - y_disp_n[0, i]) / (2*dx)
    # deformatioin gradient tensor
    f11[0, i_top] = u11[0, i_top] + 1
    f12[0, i_top] = u12[0, i_top]
    f21[0, i_top] = u21[0, i_top]
    f22[0, i_top] = u22[0, i_top] + 1
    # (lagrangian) finite strain tensor
    E11[0, i_top] = u11[0, i_top] + 0.5 * (u11[0, i_top] ** 2 + u21[0, i_top] ** 2)
    E12[0, i_top] = 0.5 * (u12[0, i_top] + u21[0, i_top]) + 0.5 * (u11[0, i_top] * u12[0, i_top] + u21[0, i_top] * u22[0, i_top])
    E21[0, i_top] = 0.5 * (u21[0, i_top] + u12[0, i_top]) + 0.5 * (u12[0, i_top] * u11[0, i_top] + u22[0, i_top] * u21[0, i_top])
    E22[0, i_top] = u22[0, i_top] + 0.5 * (u12[0, i_top] ** 2 + u22[0, i_top] ** 2)
    # invariants I
    I_E[0, i_top] = (f11[0, i_top] * f22[0, i_top] - f12[0, i_top] * f21[0, i_top])
    # invariants II
    II_E[0, i_top] = (E11[0, i_top] - (E11[0, i_top] + E22[0, i_top])/2) * (E22[0, i_top] - (E11[0, i_top] + E22[0, i_top])/2) - E21[0, i_top]**2
for i in range(nx-2):
    i_bottom = i + 1
    u11[-1, i_bottom] = (x_disp_n[-1, i+2] - x_disp_n[-1, i]) / (2*dx)
    u22[-1, i_bottom] = (y_disp_n[-1, i_bottom] - y_disp_n[-2, i_bottom]) / (dy)
    u12[-1, i_bottom] = (x_disp_n[-1, i_bottom] - x_disp_n[-2, i_bottom]) / (dy)
    u21[-1, i_bottom] = (y_disp_n[-1, i+2] - y_disp_n[-1, i]) / (2*dx)
    # deformatioin gradient tensor
    f11[-1, i_bottom] = u11[-1, i_bottom] + 1
    f12[-1, i_bottom] = u12[-1, i_bottom]
    f21[-1, i_bottom] = u21[-1, i_bottom]
    f22[-1, i_bottom] = u22[-1, i_bottom] + 1
    # (lagrangian) finite strain tensor
    E11[-1, i_bottom] = u11[-1, i_bottom] + 0.5 * (u11[-1, i_bottom] ** 2 + u21[-1, i_bottom] ** 2)
    E12[-1, i_bottom] = 0.5 * (u12[-1, i_bottom] + u21[-1, i_bottom]) + 0.5 * (u11[-1, i_bottom] * u12[-1, i_bottom] + u21[-1, i_bottom] * u22[-1, i_bottom])
    E21[-1, i_bottom] = 0.5 * (u21[-1, i_bottom] + u12[-1, i_bottom]) + 0.5 * (u12[-1, i_bottom] * u11[-1, i_bottom] + u22[-1, i_bottom] * u21[-1, i_bottom])
    E22[-1, i_bottom] = u22[-1, i_bottom] + 0.5 * (u12[-1, i_bottom] ** 2 + u22[-1, i_bottom] ** 2)
    # invariants I
    I_E[-1, i_bottom] = (f11[-1, i_bottom] * f22[-1, i_bottom] - f12[-1, i_bottom] * f21[-1, i_bottom])
    # invariants II
    II_E[-1, i_bottom] = (E11[-1, i_bottom] - (E11[-1, i_bottom] + E22[-1, i_bottom])/2) * (E22[-1, i_bottom] - (E11[-1, i_bottom] + E22[-1, i_bottom])/2) - E21[-1, i_bottom]**2

u11[u11 == 1e20] = np.nan; u22[u22 == 1e20] = np.nan; u12[u12 == 1e20] = np.nan; u21[u21 == 1e20] = np.nan
f11[f11 == 1e20] = np.nan; f22[f22 == 1e20] = np.nan; f12[f12 == 1e20] = np.nan; f21[f21 == 1e20] = np.nan
E11[E11 == 1e20] = np.nan; E21[E21 == 1e20] = np.nan; E12[E12 == 1e20] = np.nan; E22[E22 == 1e20] = np.nan
I_E[I_E == 1e20] = np.nan; II_E[II_E == 1e20] = np.nan
##################################### Bilinear interpolation from grid nodes to particles ##########################################
E11_P = np.zeros_like(undeformed_x_list); E12_P = np.zeros_like(undeformed_x_list)
E21_P = np.zeros_like(undeformed_x_list); E22_P = np.zeros_like(undeformed_x_list)
I_E_P = np.zeros_like(undeformed_x_list); II_E_P = np.zeros_like(undeformed_x_list)
Calculation = E11 ; cal = 'E11'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(undeformed_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i] 
    qx, qy = x_pp//dx , y_pp//dy + 1 
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    E11_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

Calculation = E12 ; cal = 'E12'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(undeformed_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i]
    qx, qy = x_pp//dx , y_pp//dy + 1
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    E12_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

Calculation = E21 ; cal = 'E21'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(undeformed_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i]
    qx, qy = x_pp//dx , y_pp//dy + 1
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    E21_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

Calculation = E22 ; cal = 'E22'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(disp_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i]
    qx, qy = x_pp//dx , y_pp//dy + 1
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    E22_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

Calculation = I_E ; cal = 'I_E'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(undeformed_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i]
    qx, qy = x_pp//dx , y_pp//dy + 1
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    I_E_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

Calculation = II_E ; cal = 'II_E'
Calculation = np.nan_to_num(Calculation, copy=False)
for i in range(len(undeformed_x_list)):
    x_pp, y_pp= undeformed_x_list[i], undeformed_y_list[i]
    qx, qy = x_pp//dx , y_pp//dy + 1
    qx, qy = abs(qx), abs(qy)
    qx, qy = int(qx), int(qy)
    x1, y1, f1 = X[qy, qx], Y[qy, qx], Calculation[qy, qx]
    x2, y2, f2 = X[qy - 1, qx], Y[qy - 1, qx], Calculation[qy - 1, qx]
    x3, y3, f3 = X[qy, qx + 1], Y[qy, qx + 1], Calculation[qy, qx + 1]
    x4, y4, f4 = X[qy - 1, qx + 1], Y[qy - 1, qx + 1], Calculation[qy - 1, qx + 1]
    red = f4; l_red = np.sqrt((x4-x_pp)**2 + (y4-y_pp)**2); w_red = 1/l_red**2
    blue = f3; l_blue = np.sqrt((x3-x_pp)**2 + (y3-y_pp)**2); w_blue = 1/l_blue**2
    green = f2; l_green = np.sqrt((x2-x_pp)**2 + (y2-y_pp)**2); w_green = 1/l_green**2
    yellow = f1; l_yellow = np.sqrt((x1-x_pp)**2 + (y1-y_pp)**2); w_yellow = 1/l_yellow**2
    II_E_P[i] = (red*w_red + blue*w_blue + green*w_green + yellow*w_yellow) / (w_red + w_blue + w_green + w_yellow)

print("---FDM strain calculation is done within",time.time()-start,"sec")
################################ Post processing ######################################
X_und = undeformed_cood[:,0]; Y_und = undeformed_cood[:,1]
X_ded = deformed_cood[:,0]; Y_ded = deformed_cood[:,1]
XX,YY = X_ded, Y_ded ## Select configuration for visulization (X_ded,Y_ded: deformed, X_und,Y_und: undeformed)
colormap = 'inferno'
#######################################################################################
from matplotlib.colors import Normalize
fig=plt.figure()
plt.title('horizontal_strain')
strain = E11_P
vmin = -1; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.inferno_r(abs(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='inferno', facecolors='none', vmin=vmin, vmax=vmax)
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k', linewidth=0.3)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)

fig=plt.figure()
plt.title('vertical_strain')
strain = E22_P
vmin = 0; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.inferno(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='inferno', facecolors='none', vmin=vmin, vmax=vmax)
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k', linewidth=0.3)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)

fig=plt.figure()
plt.title('shear_strain')
strain = E12_P
vmin = -1; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='seismic', facecolors='none', vmin=vmin, vmax=vmax)
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k', linewidth=0.3)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)

fig=plt.figure()
plt.title('volumetric_strain')
strain = I_E_P
vmin = -1; vmax = 3
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='seismic', facecolors='none', vmin=vmin, vmax=vmax)
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k', linewidth=0.3)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)

fig=plt.figure()
plt.title('distortioanl_strain')
strain = II_E_P
vmin = -1; vmax = 0
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.inferno(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.3)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='inferno', facecolors='none', vmin=vmin, vmax=vmax)
fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
#plt.triplot(deformed_cood[:,0],deformed_cood[:,1],ele_id, c='k', linewidth=0.3)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)
plt.show()