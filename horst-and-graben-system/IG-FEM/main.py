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
#TODO extensional model
init, extended_1km, extended_2km, extended_3km = 5814, 15930, 26049, 36249
undeform = extended_1km
deform = extended_2km
undeformed_cood = np.loadtxt("data/extension_cood0_{}.txt".format(undeform))
deformed_cood = np.loadtxt("data/extension_cood0_{}.txt".format(deform))
spin = np.loadtxt("data/angular_vel0_{}.txt".format(25188))
rad = np.loadtxt("data/radius.txt")
############################################################################################################################################################
p_num = len(undeformed_cood)
disp = deformed_cood - undeformed_cood
u_disp = disp[:,0]; v_disp = disp[:,1]
############################################################################################################################################################
# TODO triangulation
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
# TODO Solving method
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
XX,YY = X_ded, Y_ded ## select configuration for visulization (X_ded,Y_ded: deformed, X_und,Y_und: undeformed)
marker_size = 10
colormap = 'inferno'
#######################################################################################
from matplotlib.colors import Normalize
fig=plt.figure()
plt.title('horizontal_strain')
strain = solved_E11
vmin = -1; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.inferno_r(abs(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=1)
    fig.gca().add_artist(circle)
sc=plt.scatter(XX, YY, s = 0, c = strain, cmap='inferno', facecolors='none', vmin=vmin, vmax=vmax)
#fig.gca().set_xlim((-500,max(XX)+500)); fig.gca().set_ylim((-500,max(YY)+500))
fig.gca().set_xlim((16000,23000)); fig.gca().set_ylim(1975,3750)
plt.gca().set_aspect(1)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(sc, cax=cax)
plt.gcf().set_size_inches(17, 9)

fig=plt.figure()
plt.title('vertical_strain')
strain = solved_E22
vmin = -1; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.1)
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
strain = solved_E12
vmin = -1; vmax = 1
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=0.1)
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
strain = solved_volumetric
vmin = -1; vmax = 3
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.seismic(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=1)
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
strain = solved_distortion
vmin = -1; vmax = 0
norm = Normalize(vmin, vmax)
for i,j,r,k in zip(XX,YY,rad,strain):
    color = plt.cm.inferno(norm(k))
    circle = plt.Circle((i,j), r, facecolor = color, edgecolor='black', linewidth=1)
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