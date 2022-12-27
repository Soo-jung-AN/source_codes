import numpy.linalg as lina
import numpy as np
from numba import njit, float64, int64, jit
# SC_mat_e : shape func. coeff. of each element

@njit("void(float64[:,:,::1],int32[:,::1],float64[:,::1],float64[:,:,::1],int64[:,::1],float64[::1],int64)")
def M_assembly(SC_mat_e, ele_id, init_pos, PQ_detJ_e, M_RC, M_data, p_num): 
    count_sparse = 0
    TT_E = len(ele_id)

    
    for ele in range(TT_E):
        nodes = ele_id[ele]
        # x1,x2,x3 = init_pos[n1,0],init_pos[n2,0],init_pos[n3,0]
        # y1,y2,y3 = init_pos[n1,1],init_pos[n2,1],init_pos[n3,1]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        J_e = PQ_detJ_e[ele,2]


        SC_mat = SC_mat_e[ele]
        for i in range(3):
            row = nodes[i]
            c1, c2, c3 = SC_mat[0, i], SC_mat[1, i], SC_mat[2, i]
            for j in range(3):
                cc1, cc2, cc3 = SC_mat[0, j], SC_mat[1, j], SC_mat[2, j]
                col = nodes[j]
                NN = 0
                for k in range(3):
                    P = P_e[k]
                    Q = Q_e[k]
                    J = J_e[k]
                    Ni_with_gp = c1 + c2*P + c3*Q
                    Nj_with_gp = cc1 + cc2*P + cc3*Q
                    NN_with_gp = Ni_with_gp * Nj_with_gp
                    NN += 1/3 * NN_with_gp * J

                M_RC[0, count_sparse] = row
                M_RC[1, count_sparse] = col
                M_data[count_sparse] = NN
                count_sparse += 1

                M_RC[0, count_sparse] = row + p_num
                M_RC[1, count_sparse] = col + p_num
                M_data[count_sparse] = NN
                count_sparse += 1

                M_RC[0, count_sparse] = row + 2 * p_num
                M_RC[1, count_sparse] = col + 2 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1

                M_RC[0, count_sparse] = row + 3 * p_num
                M_RC[1, count_sparse] = col + 3 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1


@njit("void(float64[:,:,::1],int32[:,::1],float64[:,::1],float64[:,:,::1],int64[:,::1],float64[::1],int64)")
def A_assembly(SC_mat_e, ele_id, init_pos, PQ_detJ_e, A_RC, A_data, p_num):  
    count_sparse = 0
    TT_E = len(ele_id) #TODO 여기부터

    # TODO element for 문
    for ele in range(TT_E):
        nodes = ele_id[ele]
        # x1,x2,x3 = init_pos[n1,0],init_pos[n2,0],init_pos[n3,0]
        # y1,y2,y3 = init_pos[n1,1],init_pos[n2,1],i
        # nit_pos[n3,1]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        J_e = PQ_detJ_e[ele,2]

        SC_mat = SC_mat_e[ele]
        for i in range(3):
            row = nodes[i]
            # row_x = init_pos[i,0]
            # row_y = init_pos[i,1]

            # TODO Boundary condition
            c1, c2, c3 = SC_mat[0, i], SC_mat[1, i], SC_mat[2, i]
            for j in range(3):
                cc1, cc2, cc3 = SC_mat[0, j], SC_mat[1, j], SC_mat[2, j]
                col = nodes[j]
                NNx = 0
                NNy = 0
                for k in range(3):
                    P = P_e[k]
                    Q = Q_e[k]
                    J = J_e[k]
                    Ni_with_gp = c1 + c2*P + c3*Q
                    Nxj_with_gp = cc2
                    Nyj_with_gp = cc3
                    NNx_with_gp = Ni_with_gp * Nxj_with_gp
                    NNy_with_gp = Ni_with_gp * Nyj_with_gp
                    NNx += 1/3 * NNx_with_gp * J
                    NNy += 1/3 * NNy_with_gp * J

                A_RC[0, count_sparse] = row
                A_RC[1, count_sparse] = col
                A_data[count_sparse] = NNx
                count_sparse += 1

                A_RC[0, count_sparse] = row + p_num
                A_RC[1, count_sparse] = col + p_num
                A_data[count_sparse] = NNy
                count_sparse += 1

                A_RC[0, count_sparse] = row + 2 * p_num
                A_RC[1, count_sparse] = col + 2 * p_num
                A_data[count_sparse] = NNy
                count_sparse += 1

                A_RC[0, count_sparse] = row + 3 * p_num
                A_RC[1, count_sparse] = col + 3 * p_num
                A_data[count_sparse] = NNx
                count_sparse += 1

@njit("void(float64[:,:,::1],int32[:,::1],float64[:,::1],float64[:,:,::1],float64[::1],int64)")
def R_assembly(SC_mat_e, ele_id, init_pos, PQ_detJ_e, R_vec, p_num):
    count_sparse = 0
    TT_E = len(ele_id)

    # TODO element for 문
    for ele in range(TT_E):
        nodes = ele_id[ele]
        # x1,x2,x3 = init_pos[n1,0],init_pos[n2,0],init_pos[n3,0]
        # y1,y2,y3 = init_pos[n1,1],init_pos[n2,1],init_pos[n3,1]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        J_e = PQ_detJ_e[ele,2]

        SC_mat = SC_mat_e[ele]
        for i in range(3):
            row = nodes[i]
            c1, c2, c3 = SC_mat[0, i], SC_mat[1, i], SC_mat[2, i]
            N = 0
            for k in range(3):
                P = P_e[k]
                Q = Q_e[k]
                J = J_e[k]
                Ni_with_gp = c1 + c2*P + c3*Q
                N += 1/3*Ni_with_gp * J # weight:1/3, area:1/2
            R_vec[row] += N

            R_vec[row + p_num] += N

            #R_vec[row + 2 * p_num] += 0

            #R_vec[row + 3 * p_num] += 0
