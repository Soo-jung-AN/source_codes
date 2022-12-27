import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import numpy.linalg as lina
from Assembly import M_assembly, A_assembly, R_assembly
from preprocessing import reshape, Get_shf_coef, Get_gp_cood
from scipy import sparse
from pypardiso import spsolve
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
############################################################################################################################################################
NN = 50
L1 = 0; L2 = 30e3
H1 = 0; H2 = 10e3

def Get_deformed_cood(X,Y):
    x = 1e4 * np.cos(0.0002 * np.pi * Y) - 0.3 * X + 20000
    y = 5e3 * np.exp(-0.00001 * X) * np.cos(0.0003 * np.pi* X) - 0.1 * X + 5000
    return x,y

def Get_F(X,Y):
    result = np.array(([-0.3, -2*np.pi*np.sin(np.pi*Y/5000)],
                       [-1/20*(np.exp(-X/1e5)*(30*np.pi*np.sin(3*np.pi*X/1e4)+np.cos(3*np.pi*X/1e4)+2*np.exp(X/1e5))),0]))
    return result
############################################################################################################################################################
undeformed_cood = np.loadtxt("initial_cood(30by10km).txt") # For building unstructured mesh
rad = np.loadtxt("radius.txt")
p_num = len(undeformed_cood)
print(p_num)
deformed_cood = np.zeros((p_num, 2))
exact_E11 = np.zeros(p_num); exact_E12 = np.zeros(p_num); exact_E21 = np.zeros(p_num); exact_E22 = np.zeros(p_num)
exact_F11 = np.zeros(p_num); exact_F12 = np.zeros(p_num); exact_F21 = np.zeros(p_num); exact_F22 = np.zeros(p_num)
exact_volumetric = np.zeros(p_num); exact_distortion = np.zeros(p_num)
count=0
for i in range(p_num):
    X, Y = undeformed_cood[i]
    x, y = Get_deformed_cood(X, Y)
    deformed_cood[i] = x, y
    F = Get_F(X, Y)
    strain_tensor = 0.5*(np.dot(F.T, F) - np.eye(2))
    E11 = strain_tensor[0,0]
    E12 = strain_tensor[0,1]
    E21 = strain_tensor[1,0]
    E22 = strain_tensor[1,1]
    dev_E11 = E11 - ((E11 + E22) / 2)
    dev_E22 = E22 - ((E11 + E22) / 2)
    volumetric = (F[0, 0] * F[1, 1]) - (F[1, 0] * F[0, 1])
    distortion = dev_E11 * dev_E22 - E12 * E21
    exact_F11[count] = F[0, 0]
    exact_F12[count] = F[0, 1]
    exact_F21[count] = F[1, 0]
    exact_F22[count] = F[1, 1]
    exact_E11[count] = E11
    exact_E12[count] = E12
    exact_E21[count] = E21
    exact_E22[count] = E22
    exact_volumetric[count] = volumetric
    exact_distortion[count] = distortion
    count += 1
disp = deformed_cood - undeformed_cood
u_disp = disp[:,0]
v_disp = disp[:,1]
############################################################################################################################################################
# Triangulation
tri = Delaunay(undeformed_cood)
ele_id = tri.simplices
ele_id = reshape(undeformed_cood, ele_id, 115)
TT_E = len(ele_id); E_area = np.zeros(TT_E)
plt.figure()
plt.title("triangulated mesh")
plt.triplot(undeformed_cood[:,0],undeformed_cood[:,1],ele_id, c='k', linewidth = 0.3)
plt.gca().set_aspect(1)
plt.gcf().set_size_inches(17, 9)
#################################################################################
# Implicit-Global finite element method
import time
solving_time = time.time()
SC_mat_e = np.zeros((TT_E,3,3), dtype=np.float64)
Get_shf_coef(SC_mat_e, ele_id, undeformed_cood)
PQ_detJ_e = np.zeros((TT_E,3,3), dtype=np.float64)
Get_gp_cood(PQ_detJ_e, ele_id, undeformed_cood)

M_RC = np.zeros((2,36 * TT_E), dtype=np.int64); M_data = np.zeros(36 * TT_E, dtype=np.float64)
M_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, M_RC, M_data, p_num)
M_CSR = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num * 4, p_num * 4))

A_RC = np.zeros((2,36 * TT_E), dtype=np.int64); A_data = np.zeros(36 * TT_E, dtype=np.float64)
A_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, A_RC, A_data, p_num)
A_CSR = sparse.csr_matrix((A_data, (A_RC[0], A_RC[1])), shape=(p_num * 4, p_num * 4))

U = np.hstack((u_disp,v_disp))
U = np.hstack((U,U))

AU = A_CSR * U

R_vec = np.zeros(p_num * 4, dtype=np.float64)
R_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, R_vec, p_num)

AU_R = AU + R_vec

F = spsolve(M_CSR, AU_R)

E = 0.5 * (np.transpose(F) @ F - R_vec)
print("elasped time:",time.time()-solving_time)

solved_F11 = F[:p_num]
solved_F22 = F[p_num:2*p_num]
solved_F12 = F[2*p_num:3*p_num]
solved_F21 = F[3*p_num:]

aUaX = solved_F11 - 1
aUaY = solved_F12
aVaX = solved_F21
aVaY = solved_F22 - 1

solved_E11 = aUaX + 0.5 * (aUaX ** 2 + aVaX ** 2)
solved_E12 = 0.5 * (aUaY+aVaX) + 0.5 * (aUaX * aUaY + aVaX * aVaY)
solved_E21 = solved_E12
solved_E22 = aVaY + 0.5 * (aUaY ** 2 + aVaY ** 2)

solved_volumetric = (solved_F11 * solved_F22) - (solved_F21 * solved_F12)
trace = (solved_E11 + solved_E22) / 2
solved_distortion = ((solved_E11 - trace) * (solved_E22 - trace)) - solved_E21**2
print("---IG-FEM strain calculation is done within",time.time()-solving_time,"sec")
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
trip1 = plt.scatter(XX,YY,c=solved_E11, cmap=colormap,s = marker_size)
fig2.colorbar(trip1, ax=ax_E11)
ax_E11.set_aspect(1)
fig2.set_size_inches(15.5, 12.5)
ax_E11.set_facecolor('gray')

error_list = solved_E12 - exact_E12
ax_E12 = fig2.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX,YY,c=solved_E12, cmap="seismic",s = marker_size)
ax_E12.set_title("E12")
fig2.colorbar(trip2, ax=ax_E12)
ax_E12.set_aspect(1)
ax_E12.set_facecolor('gray')

ax_E22 = fig2.add_subplot(2, 2, 3)
trip3 = plt.scatter(XX,YY,c=solved_E22, cmap=colormap, s = marker_size)
ax_E22.set_title("E22")
fig2.colorbar(trip3, ax=ax_E22)
ax_E22.set_aspect(1)
ax_E22.set_facecolor('gray')

ax_exact = fig2.add_subplot(2, 2, 4)
ax_exact.set_title("E21")
trip4 = ax_exact.scatter(XX, YY, c=solved_E21, cmap="seismic",s = marker_size)
fig2.colorbar(trip4, ax=ax_exact)
ax_exact.set_aspect(1)

fig3 = plt.figure()
ax_volumetric = fig3.add_subplot(2, 2, 1)
ax_volumetric.set_title("volumetric")
trip1 = plt.scatter(XX, YY, c = solved_volumetric, cmap="seismic", s = marker_size)
fig3.colorbar(trip1, ax=ax_volumetric)
ax_volumetric.set_aspect(1)
fig3.set_size_inches(15.5, 12.5)
ax_volumetric.set_facecolor('gray')

ax_distortion = fig3.add_subplot(2, 2, 2)
trip2 = plt.scatter(XX, YY, c = solved_distortion, cmap=colormap, s = marker_size)
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

rmsE11=relative_root_mean_squared_error(exact_E11, solved_E11)
rmsE12=relative_root_mean_squared_error(exact_E12, solved_E12)
rmsE22=relative_root_mean_squared_error(exact_E22, solved_E22)
rmsJ=relative_root_mean_squared_error(exact_volumetric, solved_volumetric)
rmsII_E=relative_root_mean_squared_error(exact_distortion, solved_distortion)
print("#############")
print("IG-FEM RRRMS_ of E11 : ",rmsE11)
print("IG-FEM RRRMS_ of E12 : ",rmsE12)
print("IG-FEM RRRMS_ of E22 : ",rmsE22)
print("IG-FEM RRRMS_ of volumetric : ", rmsJ)
print("IG-FEM RRRMS_ of distortion : ", rmsII_E)
#################################################################################################################
plt.show()