
from msilib.schema import Condition
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy.linalg as lina
import time
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
########################################################################################
L1 = 0; L2 = 30e3
H1 = 0; H2 = 13e3 # 3 km spared space on the top for padding
nx = 60; ny = 25 # Selcting Grid resolution
########################################################################################
X_lin = np.linspace(L1,L2,nx)
Y_lin = np.linspace(H1,H2,ny)
X,Y = np.meshgrid(X_lin,Y_lin)

def Get_deformed_cood(X,Y):
    x = 1e4 * np.cos(0.0002 * np.pi * Y) - 0.3 * X + 20000
    y = 5e3 * np.exp(-0.00001 * X) * np.cos(0.0003 * np.pi* X) - 0.1 * X + 5000
    
    return x,y

def Get_F(X,Y):
    result = np.array(([-0.3, -2*np.pi*np.sin(np.pi*Y/5000)],
                       [-1/20*(np.exp(-X/1e5)*(30*np.pi*np.sin(3*np.pi*X/1e4)+np.cos(3*np.pi*X/1e4)+2*np.exp(X/1e5))),0]))
    return result
########################################################################################
undeformed_cood = np.loadtxt("initial_cood(30by10km).txt")
rad = np.loadtxt("radius.txt")
p_num = len(undeformed_cood)
deformed_cood = np.zeros_like(undeformed_cood)
deformed_x_list = deformed_cood[:, 0]; deformed_y_list = deformed_cood[:, 1]
undeformed_x_list = undeformed_cood[:,0]; undeformed_y_list = undeformed_cood[:,1]
exact_E11 = np.zeros(len(undeformed_x_list)); exact_E12 = np.zeros(len(undeformed_x_list))
exact_E21 = np.zeros(len(undeformed_x_list)); exact_E22 = np.zeros(len(undeformed_x_list))
exact_J = np.zeros(len(undeformed_x_list)); exact_II_E = np.zeros(len(undeformed_x_list))
count = 0
for i in range(len(undeformed_x_list)):
    # Transformating centroids of particles using defined "Get_deformed_cood" function
    undeformed_x = undeformed_x_list[i]
    undeformed_y = undeformed_y_list[i]
    result = Get_deformed_cood(undeformed_x, undeformed_y)
    deformed_x_list[i] = result[0]
    deformed_y_list[i] = result[1]

    # Calculating analytical deformation gradient tensor (F) and Green strain tensor (E)
    F = Get_F(undeformed_x, undeformed_y)
    green_strain_tensor = 0.5*(np.dot(F.T, F) - np.eye(2))
    E11 = green_strain_tensor[0,0]
    E12 = green_strain_tensor[0,1]
    E21 = green_strain_tensor[1,0]
    E22 = green_strain_tensor[1,1]
    dev_E11 = E11 - ((E11 + E22) / 2)
    dev_E22 = E22 - ((E11 + E22) / 2)
    J = F[0, 0] * F[1, 1] - F[1, 0] * F[0, 1]
    II_E = dev_E11 * dev_E22 - E12 * E21
    exact_E11[count] = E11
    exact_E12[count] = E12
    exact_E21[count] = E21
    exact_E22[count] = E22
    exact_J[count] = J
    exact_II_E[count] = II_E
    count += 1

disp = deformed_cood - undeformed_cood
disp_x_list = disp[:,0]
disp_y_list = disp[:,1]
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

# for i in range(nx-2): # we blocked 
#     i_bottom = i + 1
#     u11[-1, i_bottom] = (x_disp_n[-1, i+2] - x_disp_n[-1, i]) / (2*dx)
#     u22[-1, i_bottom] = (y_disp_n[-1, i_bottom] - y_disp_n[-2, i_bottom]) / (dy)
#     u12[-1, i_bottom] = (x_disp_n[-1, i_bottom] - x_disp_n[-2, i_bottom]) / (dy)
#     u21[-1, i_bottom] = (y_disp_n[-1, i+2] - y_disp_n[-1, i]) / (2*dx)
#     # deformatioin gradient tensor
#     f11[-1, i_bottom] = u11[-1, i_bottom] + 1
#     f12[-1, i_bottom] = u12[-1, i_bottom]
#     f21[-1, i_bottom] = u21[-1, i_bottom]
#     f22[-1, i_bottom] = u22[-1, i_bottom] + 1
#     # (lagrangian) finite strain tensor
#     E11[-1, i_bottom] = u11[-1, i_bottom] + 0.5 * (u11[-1, i_bottom] ** 2 + u21[-1, i_bottom] ** 2)
#     E12[-1, i_bottom] = 0.5 * (u12[-1, i_bottom] + u21[-1, i_bottom]) + 0.5 * (u11[-1, i_bottom] * u12[-1, i_bottom] + u21[-1, i_bottom] * u22[-1, i_bottom])
#     E21[-1, i_bottom] = 0.5 * (u21[-1, i_bottom] + u12[-1, i_bottom]) + 0.5 * (u12[-1, i_bottom] * u11[-1, i_bottom] + u22[-1, i_bottom] * u21[-1, i_bottom])
#     E22[-1, i_bottom] = u22[-1, i_bottom] + 0.5 * (u12[-1, i_bottom] ** 2 + u22[-1, i_bottom] ** 2)
#     # invariants I
#     I_E[-1, i_bottom] = (f11[-1, i_bottom] * f22[-1, i_bottom] - f12[-1, i_bottom] * f21[-1, i_bottom])
#     # invariants II
#     II_E[-1, i_bottom] = (E11[-1, i_bottom] - (E11[-1, i_bottom] + E22[-1, i_bottom])/2) * (E22[-1, i_bottom] - (E11[-1, i_bottom] + E22[-1, i_bottom])/2) - E21[-1, i_bottom]**2

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
print("---post-processing...")
X_und = undeformed_cood[:,0]; Y_und = undeformed_cood[:,1]
X_ded = deformed_cood[:,0]; Y_ded = deformed_cood[:,1]
XX,YY = X_ded, Y_ded ## Select configuration for visulization (X_ded,Y_ded: deformed, X_und,Y_und: undeformed)
marker_size = 10
colormap = 'inferno'
#######################################################################################
fig2 = plt.figure()
ax_E11 = fig2.add_subplot(2, 2, 1)
ax_E11.set_title("E11")
trip1 = plt.scatter(XX,YY,c=E11_P, cmap=colormap,s = marker_size)
fig2.colorbar(trip1, ax=ax_E11)
ax_E11.set_aspect(1)
fig2.set_size_inches(15.5, 12.5)
ax_E11.set_facecolor('gray')

error_list = E12_P - exact_E12
ax_E12 = fig2.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX,YY,c=E12_P, cmap="seismic",s = marker_size)
ax_E12.set_title("E12")
fig2.colorbar(trip2, ax=ax_E12)
ax_E12.set_aspect(1)
ax_E12.set_facecolor('gray')

ax_E22 = fig2.add_subplot(2, 2, 3)
trip3 = plt.scatter(XX,YY,c=E22_P, cmap=colormap, s = marker_size)
ax_E22.set_title("E22")
fig2.colorbar(trip3, ax=ax_E22)
ax_E22.set_aspect(1)
ax_E22.set_facecolor('gray')

ax_exact = fig2.add_subplot(2, 2, 4)
ax_exact.set_title("E21")
trip4 = ax_exact.scatter(XX, YY, c=E21_P, cmap="seismic",s = marker_size)
fig2.colorbar(trip4, ax=ax_exact)
ax_exact.set_aspect(1)

fig3 = plt.figure()
ax_volumetric = fig3.add_subplot(2, 2, 1)
ax_volumetric.set_title("volumetric")
trip1 = plt.scatter(XX, YY, c = I_E_P, cmap="seismic", s = marker_size)
fig3.colorbar(trip1, ax=ax_volumetric)
ax_volumetric.set_aspect(1)
fig3.set_size_inches(15.5, 12.5)
ax_volumetric.set_facecolor('gray')

ax_distortion = fig3.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX, YY, c = II_E_P, cmap=colormap, s = marker_size)
ax_distortion.set_title("distiortion")
fig3.colorbar(trip2, ax=ax_distortion)
ax_distortion.set_aspect(1)
ax_distortion.set_facecolor('gray')

####################################### Relative Root Mean Square Error #########################################
def relative_root_mean_squared_error(true, pred):
    size = len(true)
    num = np.sqrt(np.sum(np.square(true - pred))/size)
    den = np.sum(abs(true))/size
    rrmse_loss = num/den
    return rrmse_loss * 100

rmsE11=relative_root_mean_squared_error(exact_E11, E11_P)
rmsE12=relative_root_mean_squared_error(exact_E12, E12_P)
rmsE22=relative_root_mean_squared_error(exact_E22, E22_P)
rmsJ=relative_root_mean_squared_error(exact_J, I_E_P)
rmsII_E=relative_root_mean_squared_error(exact_II_E, II_E_P)
print("#############")
print("FDM RRRMS_ of E11 : ",rmsE11)
print("FDM RRRMS_ of E12 : ",rmsE12)
print("FDM RRRMS_ of E22 : ",rmsE22)
print("FDM RRRMS_ of volumetric : ", rmsJ)
print("FDM RRRMS_ of distortion : ", rmsII_E)
#################################################################################################################
plt.show()
