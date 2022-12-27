import numpy.linalg as lina
import numpy as np
from numba import njit, float64, int64, jit


def reshape(init_pos, ele_id, alpha):
    del_id = []
    # for i in range(len(ele_id)):
    #     id_tri_temp = ele_id[i]
    #     a,b,c = init_pos[id_tri_temp]
    #     length1 = np.sqrt(np.sum((a-b)**2))
    #     length2 = np.sqrt(np.sum((a-c)**2))
    #     length3 = np.sqrt(np.sum((b-c)**2))
    #     if length1 > alpha or length2 > alpha or length3 > alpha:
    #         del_id.append(i)
    # return np.delete(ele_id,del_id,0)

def reshape(undeformed_cood, ele_id, alpha): # alpha is maximum diameter, 3이면 maximum rad의 3배까진 봐주겟다 이거임
    del_id = []
    for i in range(len(ele_id)):
        e1,e2,e3 = ele_id[i]
        x1,y1 = undeformed_cood[e1]
        x2,y2 = undeformed_cood[e2]
        x3,y3 = undeformed_cood[e3]
        l1 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        l2 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
        l3 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
        angle1 = np.degrees(np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2)))
        angle2 = np.degrees(np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3)))
        angle3 = np.degrees(np.arccos((l2**2 + l3**2 - l1**2)/(2*l2*l3)))
        if (angle1>alpha) or (angle2>alpha) or (angle3>alpha):
            del_id.append(i)
    return np.delete(ele_id, del_id, 0)


@njit("void(float64[:,:,::1], int32[:,::1], float64[:,::1])")
def Get_shf_coef(SC_mat_e, ele_id, init_pos): # 임의로 shape function coefficient matrix라 명명
    Base3x3 = np.zeros((3,3), dtype=np.float64)
    for ele in range(len(ele_id)):
        n1,n2,n3 = ele_id[ele]
        x1,x2,x3 = init_pos[n1,0],init_pos[n2,0],init_pos[n3,0]
        y1,y2,y3 = init_pos[n1,1],init_pos[n2,1],init_pos[n3,1]
        Base3x3[0, 0], Base3x3[0, 1], Base3x3[0, 2] = 1, x1, y1
        Base3x3[1, 0], Base3x3[1, 1], Base3x3[1, 2] = 1, x2, y2
        Base3x3[2, 0], Base3x3[2, 1], Base3x3[2, 2] = 1, x3, y3
        SC_mat = lina.inv(Base3x3)
        for i in range(3):
            for j in range(3):
                SC_mat_e[ele, i, j] = SC_mat[i, j]


@njit("void(float64[:,:,::1],int32[:,::1],float64[:,::1])")
def Get_gp_cood(PQ_detJ_e, ele_id, init_pos):
    s_list = np.array([1/6, 2/3, 1/6], dtype=float64)
    t_list = np.array([1/6, 1/6, 2/3], dtype=float64)
    for ele in range(len(ele_id)):
        n1,n2,n3 = ele_id[ele]
        x1,x2,x3 = init_pos[n1,0],init_pos[n2,0],init_pos[n3,0]
        y1,y2,y3 = init_pos[n1,1],init_pos[n2,1],init_pos[n3,1]
        for i in range(3):
            s = s_list[i]
            t = t_list[i]
            N1 = 1-s-t
            N2 = s
            N3 = t
            P = x1 * N1 + x2 * N2 + x3 * N3
            Q = y1 * N1 + y2 * N2 + y3 * N3
            PQ_detJ_e[ele,0,i] = P
            PQ_detJ_e[ele,1,i] = Q
            J11 = -x1 + x2  # dP/ds
            J12 = -x1 + x3  # dP/dt
            J21 = -y1 + y2 # dQ/ds
            J22 = -y1 + y3  # dQ/dt
            det_J = J11 * J22 - J12 * J21
            PQ_detJ_e[ele,2,i] = det_J # TODO 아마도 element당 하나 값만 있으면 될듯? 확실치 않음

