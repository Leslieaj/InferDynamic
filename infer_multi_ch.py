import numpy as np
from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
import math
# import random

from dynamics import dydx1, dydx2, fvdp2, fvdp2_1, ode_test
from generator import generate_complete_polynomail
from draw import draw, draw2D

def simulation_ode(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0):
    """ Given a ODE function, some initial state, stepsize, then return the points.
        @ode_func: ODE function 
        @y0: inital state
        @t_tuple: 
        @stepsize: step size
        @eps: guass noise (defult 0)
    """

    t_list = []
    y_list = []

    for k in range(0,len(y0)):
        t_start = t_tuple[k][0]
        t_end = t_tuple[k][1]
        t_points = np.arange(t_start, t_end + stepsize, stepsize)
        y_object = solve_ivp(ode_func, (t_start, t_end+stepsize), y0[k], t_eval = t_points, rtol=1e-7, atol=1e-9)
        y_points = y_object.y.T
        if eps > 0:
            for i in range(0,y_points.shape[0]):
                for j in range(0,y_points.shape[1]):
                    y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
        t_list.append(t_points)
        y_list.append(y_points)
    return t_list, y_list

def diff_method(t_list, y_list, order, stepsize):
    """Using multi-step difference method (Adams5) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomail(L_y,order)
    L_p = gene.shape[0]

    

    for k in range(0,len(y_list)):
        t_points = t_list[k]
        L_t = len(t_points)
        D = L_t - 4    #Adams5
        y_points = y_list[k]
        A_matrix = np.zeros((D*2, L_p), dtype=np.double)
        b_matrix = np.zeros((D*2, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        # Adams5
        for i in range(0, 2*D):
            if i < D:     # forward
                A_matrix[i] = (-19*coef_matrix[i]+106*coef_matrix[i+1]-264*coef_matrix[i+2]+646*coef_matrix[i+3]+251*coef_matrix[i+4])*stepsize/720
                b_matrix[i] = y_points[i+4]-y_points[i+3]
            else:         # backward
                A_matrix[i] = (-19*coef_matrix[i-D+4]+106*coef_matrix[i-D+3]-264*coef_matrix[i-D+2]+646*coef_matrix[i-D+1]+251*coef_matrix[i-D])*stepsize/720
                b_matrix[i] = y_points[i-D+1]-y_points[i-D]
        
        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]

    return final_A_mat, final_b_mat

def infer_dynamic(t_list, y_list, stepsize, order):
    """ The main function to infer a dynamic system.
    """
    start = time.time()
    A, b = diff_method(t_list, y_list, order, stepsize)
    end_diff = time.time()
    # Moore-Penrose Inverse (pseudoinverse)
    # g = linalg.pinv(A).dot(b) # using least-square solver
    g = pinv2(A).dot(b) # using using its singular-value decomposition and including all 'large' singular values.
    end_pseudoinv = time.time()
    return g.T, end_diff-start, end_pseudoinv-end_diff


def parti(t_list,y_list,ep=0.5,an=1/6):
    tpar_list = []
    ypar_list = []
    for l in range(0,len(y_list)):
        t_points = t_list[l]
        y_points = y_list[l]
        stepsize = t_list[l][1] - t_list[l][0]
        row = y_points.shape[0]
        col = y_points.shape[1]
        diffmat = (y_points[1:][:] - y_points[0:row-1][:])/stepsize
        parpo = [0]
        diffmat2 = np.multiply(diffmat,diffmat)
        diffvecm = diffmat2.sum(axis=1)
        for i in range(0,row-2):
            parbool = 0
            if diffvecm[i+1]/diffvecm[i] > 1+ep or diffvecm[i+1]/diffvecm[i] < 1-ep:
                parbool = 1
            if np.dot(diffmat[i+1],diffmat[i].T)/math.sqrt(diffvecm[i+1]*diffvecm[i]) < np.sin(an*np.pi):
                parbool = 1
            if parbool == 1 and parpo[-1] != i-1:
                parpo.append(i)
        parpo.append(row-1)
        for i in range(0,len(parpo)-1):
            tpar_list.append(t_points[parpo[i]:parpo[i+1]+1])
            ypar_list.append(y_points[parpo[i]:parpo[i+1]+1][:])
    return tpar_list, ypar_list


def dist(y_list,y_list_test):
    leny = len(y_list)
    g = 0
    for i in range(0,leny):
        if y_list[i].shape[0] == y_list_test[i].shape[0] and y_list[i].shape[1] == y_list_test[i].shape[1]:
            mat = y_list[i] - y_list_test[i]
            mat = mat**2
            g = max(g,mat.max())
            g = math.sqrt(g)
        else:
            g = 9999
    return g
        

def infer_dynamic_modes(t_list, y_list, stepsize, maxorder, ep=0.01):
    len_tr = len(y_list)
    modes = []
    coefs = []
    for l in range(0,len_tr):
        dis = []
        cgt = []
        for i in range(0,len(modes)):
            comt = []
            comy = []
            y0 = []
            t_tuple = []
            for j in range(0,len(modes[i])):
                comt.append(t_list[modes[i][j]])
                comy.append(y_list[modes[i][j]])
                y0.append(y_list[modes[i][j]][0])
                t_tuple.append((t_list[modes[i][j]][0],t_list[modes[i][j]][-1]))
            comt.append(t_list[l])
            comy.append(y_list[l])
            y0.append(y_list[l][0])
            t_tuple.append((t_list[l][0],t_list[l][-1]))
            A, b = diff_method(comt, comy, maxorder, stepsize)
            g = pinv2(A).dot(b)
            comttest, comytest = simulation_ode(ode_test(g.T,maxorder), y0, t_tuple, stepsize, eps=0)
            dis.append(dist(comy,comytest))
            cgt.append(g.T)
        if len(modes) == 0:
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], maxorder, stepsize)
            g = pinv2(A).dot(b)
            modes.append([l])
            coefs.append(g.T)
        elif min(dis) >= ep:
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], maxorder, stepsize)
            g = pinv2(A).dot(b)
            modes.append([l])
            coefs.append(g.T)
        else:
            modes[dis.index(min(dis))].append(l)
            coefs[dis.index(min(dis))] = cgt[dis.index(min(dis))]

    return modes, coefs

def infer_dynamic_modes_ex(t_list, y_list, stepsize, maxorder, ep=0.01):
    len_tr = len(y_list)
    modes = []
    coefs = []
    mdors = []
    for l in range(0,len_tr):
        dis = []
        for i in range(0,len(modes)):
            t_list_l, y_list_l = simulation_ode(ode_test(coefs[i],mdors[i]), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
            dis.append(dist(y_list[l:l+1],y_list_l))

        if len(modes) == 0:
            dis = []
            cgt = []
            for order in range(0,maxorder+1):
                A, b = diff_method(t_list[l:l+1], y_list[l:l+1], order, stepsize)
                g = pinv2(A).dot(b)
                t_list_l, y_list_l = simulation_ode(ode_test(g.T,order), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
                dis.append(dist(y_list[l:l+1],y_list_l))
                cgt.append(g.T)
            od = dis.index(min(dis))
            modes.append([l])
            coefs.append(cgt[od])
            mdors.append(od)
        elif min(dis) >= ep:
            dis = []
            cgt = []
            for order in range(0,maxorder+1):
                A, b = diff_method(t_list[l:l+1], y_list[l:l+1], order, stepsize)
                g = pinv2(A).dot(b)
                t_list_l, y_list_l = simulation_ode(ode_test(g.T,order), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
                dis.append(dist(y_list[l:l+1],y_list_l))
                cgt.append(g.T)
            od = dis.index(min(dis))
            modes.append([l])
            coefs.append(cgt[od])
            mdors.append(od)
        else:
            od = dis.index(min(dis))
            comt = []
            comy = []
            y0 = []
            t_tuple = []
            for j in range(0,len(modes[od])):
                comt.append(t_list[modes[od][j]])
                comy.append(y_list[modes[od][j]])
                y0.append(y_list[modes[od][j]][0])
                t_tuple.append((t_list[modes[od][j]][0],t_list[modes[od][j]][-1]))
            comttest, comytest = simulation_ode(ode_test(coefs[od],mdors[od]), y0, t_tuple, stepsize, eps=0)
            d1 = dist(comy,comytest)
            comt.append(t_list[l])
            comy.append(y_list[l])
            y0.append(y_list[l][0])
            t_tuple.append((t_list[l][0],t_list[l][-1]))
            A, b = diff_method(comt, comy, mdors[od], stepsize)
            g = pinv2(A).dot(b)
            comttest, comytest = simulation_ode(ode_test(coefs[od],mdors[od]), y0, t_tuple, stepsize, eps=0)
            d2 = dist(comy,comytest)
            modes[od].append(l)
            if d2 < d1 :
                coefs[od] = g.T


    return modes, coefs, mdors

if __name__ == "__main__":
    y0 = [[-1],[5],[1],[3],[0],[2],[-2]]
    t_tuple = [(0,4),(0,4),(0,4),(0,4),(0,4),(0,4),(0,4),(0,4)]
    stepsize = 0.01
    order = 6
    maxorder = 6
    # start = time.time()
    t_list, y_list = simulation_ode(dydx1, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    modes, coefs = infer_dynamic_modes(t_list, y_list, stepsize, maxorder,0.1)
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()
    print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)
    print(modes,coefs)
    for i in range(0,len(y_list)):
        plt.plot(t_list[i], y_list[i])
    plt.show()
    
