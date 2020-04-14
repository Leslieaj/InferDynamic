import numpy as np
from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
import math
import random

from dynamics import dydx1, dydx2, fvdp2, fvdp2_1, ode_test
from generator import generate_complete_polynomial
from draw import draw, draw2D

import sklearn.cluster as skc
from sklearn import metrics   # 评估模型
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model
from sklearn.linear_model import Ridge
from libsvm.svmutil import *



def simulation_ode(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0, solve_method='RK45'):
    """ Given a ODE function, some initial state, stepsize, then return the points.

        Use RK45 method for solving ODE.

        @ode_func: ODE function 
        @y0: inital state
        @t_tuple: 
        @stepsize: step size
        @noise_type: type of noise (e.g. 1 for Gaussian)
        @eps: guass noise (default 0 for no noise)

        Returns:
        @t_list: list of list of time points.
        @y_list: list of points at each time point.

    """

    t_list = []
    y_list = []

    for k in range(0,len(y0)):
        t_start = t_tuple[k][0]
        t_end = t_tuple[k][1]
        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        y_object = solve_ivp(ode_func, (t_start, t_end + 1.1*stepsize), y0[k], t_eval = t_points, method=solve_method, rtol=1e-7, atol=1e-9, max_step = stepsize/100)
        y_points = y_object.y.T
        if eps > 0:
            for i in range(0,y_points.shape[0]):
                for j in range(0,y_points.shape[1]):
                    y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
        t_list.append(t_points)
        y_list.append(y_points)
    return t_list, y_list


def simulation_ode_2(modelist, event, y0, t_tuple, stepsize):
    
    t_list = []
    y_list = []

    for k in range(0,len(y0)):
        t_start = t_tuple[k][0]
        t_end = t_tuple[k][1]
        yinitial = y0[k]

        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        T_points = np.linspace(t_start, t_end, num)
        # change = 1
        status = 1
        if event(0,y0[k])>0:
            label = 0
        else:
            label = 1
        ite = 0
        # while change > 0:
        while status == 1:
            ite = ite + 1
            print(ite)
            print(yinitial)
            y_object = solve_ivp(modelist[label], (t_start, t_end + 1.1*stepsize), yinitial, t_eval = t_points, method='RK45', rtol=1e-7, atol=1e-9, max_step = stepsize/10, events=[event], dense_output=True)
            # change = y_object.t_events[0].shape[0]
            status = y_object.status
            if status == 1:
                if label == 1:
                    label = 0
                else:
                    label = 1
                t_start = y_object.t_events[0][0]
                print(t_start)
                l = t_points.shape[0]
                for i in range(0,l):
                    if t_points[i] > t_start:
                        break
                t_points=t_points[i:]
                sol = y_object.sol
                yinitial = sol.__call__(t_start + stepsize/100)
                y_points = y_object.y.T
                y_points = y_points[0:i]
            else: 
                y_points = y_object.y.T

            if ite == 1:
                Y_points = y_points
            else:
                Y_points = np.r_[Y_points,y_points]
        t_list.append(T_points)
        y_list.append(Y_points)
    return t_list, y_list



def simulation_ode_3(modelist, eventlist, labelfun, y0, t_tuple, stepsize):
    
    t_list = []
    y_list = []

    for k in range(0,len(y0)):
        t_start = t_tuple[k][0]
        t_end = t_tuple[k][1]
        yinitial = y0[k]

        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        T_points = np.linspace(t_start, t_end, num)
        # change = 1
        status = 1
        label = labelfun(y0[k])
        # while change > 0:
        ite = 0
        while status == 1:
            ite = ite + 1
            # print(ite)
            # print(yinitial)
            y_object = solve_ivp(modelist[label], (t_start, t_end + 1.1*stepsize), yinitial, t_eval = t_points, method='RK45', rtol=1e-7, atol=1e-9, max_step = stepsize/10, events=[eventlist[label]], dense_output=True)
            # change = y_object.t_events[0].shape[0]
            status = y_object.status
            if status == 1:
                label = label+1
                if label>2:
                    label = 0
                t_start = y_object.t_events[0][0]
                print(t_start)
                l = t_points.shape[0]
                for i in range(0,l):
                    if t_points[i] > t_start:
                        break
                t_points=t_points[i:]
                sol = y_object.sol
                yinitial = sol.__call__(t_start + stepsize/100)
                y_points = y_object.y.T
                y_points = y_points[0:i]
            else: 
                y_points = y_object.y.T

            if ite == 1:
                Y_points = y_points
            else:
                Y_points = np.r_[Y_points,y_points]
        t_list.append(T_points)
        y_list.append(Y_points)
    return t_list, y_list

# def simulation_ode_stiff(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0):
#     """ Given a ODE function, some initial state, stepsize, then return the points.


#         @ode_func: ODE function 
#         @y0: inital state
#         @t_tuple: 
#         @stepsize: step size
#         @eps: guass noise (defult 0)
#     """

#     t_list = []
#     y_list = []

#     for k in range(0,len(y0)):
#         t_start = t_tuple[k][0]
#         t_end = t_tuple[k][1]
#         num = round((t_end - t_start)/stepsize + 1)
#         t_points = np.linspace(t_start, t_end, num)
#         y_object = solve_ivp(ode_func, (t_start, t_end + 1.1*stepsize), y0[k], t_eval = t_points, method ='BDF', rtol=1e-7, atol=1e-9)
#         y_points = y_object.y.T
#         if eps > 0:
#             for i in range(0,y_points.shape[0]):
#                 for j in range(0,y_points.shape[1]):
#                     y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
#         t_list.append(t_points)
#         y_list.append(y_points)
#     return t_list, y_list

def norm(V):
    s = 0
    for i in range(0,V.shape[0]):
        s= s+ V[i]**2
    return np.sqrt(s)

def diff_method(t_list, y_list, order, stepsize):
    """Using multi-step difference method (Adams5) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)
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


def diff_method1(t_list, y_list, order, stepsize):
    """Using multi-step difference method (backward differential formula)
    to compute the estimated derivative at each point.

    @t_list: list of time points.
    @y_list: list of points at every time point.
    @order: 

    """
    final_A_mat = None
    final_b_mat = None
    
    # Number of variables
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)

    # Number of monomials
    L_p = gene.shape[0]

    # For every trajectory
    for k in range(len(y_list)):
        t_points = t_list[k]
        y_points = y_list[k]

        # L_t: number of total time points. D: number of points we will get
        L_t = len(t_points)
        D = L_t - 5    #bdf5
        if D < 1:
            continue

        # print(D,L_p)

        # A_matrix = np.zeros((D*2, L_p), dtype=np.double)
        # b_matrix = np.zeros((D*2, L_y),  dtype=np.double)
        A_matrix = np.zeros((D, L_p), dtype=np.double)
        b_matrix = np.zeros((D, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        # for i in range(0, 2*D):
        #     if i < D:     # forward
        #         A_matrix[i] = 60 * stepsize * coef_matrix[i+5]
        #         b_matrix[i] = 137 * y_points[i+5] - 300 * y_points[i+4] + 300 * y_points[i+3] - 200 * y_points[i+2] + 75 * y_points[i+1] - 12 * y_points[i]
        #         A_matrix[i] = (-19*coef_matrix[i]+106*coef_matrix[i+1]-264*coef_matrix[i+2]+646*coef_matrix[i+3]+251*coef_matrix[i+4])*stepsize/720
        #         b_matrix[i] = y_points[i+4]-y_points[i+3]
        #     else:         # backward
        #         A_matrix[i] = 60 * stepsize * coef_matrix[i-D]
        #         b_matrix[i] = 137 * y_points[i-D] - 300 * y_points[i-D+1] + 300 * y_points[i-D+2] - 200 * y_points[i-D+3] + 75 * y_points[i-D+4] - 12 * y_points[i-D+5]
        #         A_matrix[i] = (-19*coef_matrix[i-D+4]+106*coef_matrix[i-D+3]-264*coef_matrix[i-D+2]+646*coef_matrix[i-D+1]+251*coef_matrix[i-D])*stepsize/720
        #         b_matrix[i] = y_points[i-D+1]-y_points[i-D]
        
        for i in range(0, D):
            A_matrix[i] = coef_matrix[i+5]
            b_matrix[i] = (137 * y_points[i+5] - 300 * y_points[i+4] + 300 * y_points[i+3] - 200 * y_points[i+2] + 75 * y_points[i+1] - 12 * y_points[i])/(60 * stepsize)

        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
        else:
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]

    return final_A_mat, final_b_mat


def infer_dynamic(t_list, y_list, stepsize, order):
    """Infer the ODE from a trajectory assumed to be in a single mode.

    Uses pseudoinverse computation.

    Returns the coefficients of the ODE.

    """
    start = time.time()
    A, b = diff_method(t_list, y_list, order, stepsize)
    end_diff = time.time()

    # Moore-Penrose Inverse (pseudoinverse)
    # g = linalg.pinv(A).dot(b)  # using least-square solver

    # Using its singular-value decomposition and including all 'large' singular values.
    # Faster than pinv.
    g = pinv2(A).dot(b)
    end_pseudoinv = time.time()
    return g.T, end_diff-start, end_pseudoinv-end_diff


def parti(t_list, y_list, ep=0.5, an=1/6):
    """Partition method.

    @an: angle.

    """
    tpar_list = []
    ypar_list = []

    for l in range(len(y_list)):
        t_points = t_list[l]
        y_points = y_list[l]
        stepsize = t_list[l][1] - t_list[l][0]

        # Number of data points
        row = y_points.shape[0]

        # Number of variables
        col = y_points.shape[1]

        # Difference between consecutive points
        diffmat = (y_points[1:][:] - y_points[0:row-1][:])/stepsize

        # parpo stands for partition position
        parpo = [-1]

        # Find sum of square of difference at each point
        diffmat2 = np.multiply(diffmat,diffmat)
        diffvecm = diffmat2.sum(axis=1)

        for i in range(row-2):
            parbool = 0
            # Detect of sum of squares changes by a lot
            if diffvecm[i+1]/diffvecm[i] > 1+ep or diffvecm[i+1]/diffvecm[i] < 1-ep:
                parbool = 1
            # If angle between consecutive derivatives is larger than limit
            if np.dot(diffmat[i+1],diffmat[i].T)/math.sqrt(diffvecm[i+1]*diffvecm[i]) < np.sin(an*np.pi):
                parbool = 1
            # Then start a new mode (if not at start of a mode already)
            if parbool == 1 and parpo[-1] != i:
                parpo.append(i+1)
        parpo.append(row-1)
        # Add the rest
        for i in range(0,len(parpo)-1):
            if parpo[i+1] - parpo[i]+1 > 5:
                tpar_list.append(t_points[parpo[i]+1:parpo[i+1]])
                ypar_list.append(y_points[parpo[i]+1:parpo[i+1]][:])

    return tpar_list, ypar_list


def dist(y_list, y_list_test):
    """Maximum distance between two lists of matrices.

    """
    leny = len(y_list)
    g = 0
    for i in range(0,leny):
        if y_list[i].shape[0] == y_list_test[i].shape[0] and y_list[i].shape[1] == y_list_test[i].shape[1]:
            mat = y_list[i] - y_list_test[i]
            mat = np.fabs(mat)
            g = np.max(mat)
        else:
            g = 9999
    return g
        

def infer_dynamic_modes(t_list, y_list, stepsize, maxorder, ep=0.01):
    """Implementation of a clustering method.

    Later methods (infer_dynamic_modes_ex) should be better.

    """
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
    """Implementation of a clustering method.

    """
    len_tr = len(y_list)
    mode = []
    mode_tls = []
    mode_yls = []
    mode_y0l = []
    mode_tul = []
    mode_coe = []
    mode_ord = []
    for l in range(0,len_tr):
        print("trace"+str(l))
        
        for od in range(0,maxorder):
            print("try of order"+str(od))
            dis = []
            cgt = []
            for i in range(0,len(mode)):
                comt = mode_tls[i].copy()
                comy = mode_yls[i].copy()
                comy0 = mode_y0l[i].copy()
                comtu = mode_tul[i].copy()
                comt.append(t_list[l])
                comy.append(y_list[l])
                comy0.append(y_list[l][0])
                comtu.append((t_list[l][0],t_list[l][-1]))
                A, b = diff_method(comt, comy, od, stepsize)
                g = pinv2(A).dot(b)
                comttest, comytest = simulation_ode(ode_test(g.T,od), comy0, comtu, stepsize, eps=0, solve_method='BDF')
                dis.append(dist(comy,comytest))
                cgt.append(g.T)
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], od, stepsize)
            g = pinv2(A).dot(b)
            t_list_l, y_list_l = simulation_ode(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0, solve_method='BDF')
            dis.append(dist(y_list[l:l+1],y_list_l))
            cgt.append(g.T)
            if min(dis) < ep:
                if len(mode) == 0:
                    p = len(mode)
                    mode.append([l])
                    mode_tls.append([t_list[l]])
                    mode_yls.append([y_list[l]])
                    mode_y0l.append([y_list[l][0]])
                    mode_tul.append([(t_list[l][0],t_list[l][-1])])
                    mode_coe.append(cgt[p])
                    mode_ord.append(od)
                elif min(dis[0:len(mode)]) < ep:
                    p = dis[0:len(mode)].index(min(dis[0:len(mode)]))
                    mode[p].append(l)
                    mode_tls[p].append(t_list[l])
                    mode_yls[p].append(y_list[l])
                    mode_y0l[p].append(y_list[l][0])
                    mode_tul[p].append((t_list[l][0],t_list[l][-1]))
                    mode_ord[p] = od
                    mode_coe[p] = cgt[p]
                else:
                    p = len(mode)
                    mode.append([l])
                    mode_tls.append([t_list[l]])
                    mode_yls.append([y_list[l]])
                    mode_y0l.append([y_list[l][0]])
                    mode_tul.append([(t_list[l][0],t_list[l][-1])])
                    mode_coe.append(cgt[p])
                    mode_ord.append(od)
                break

    return mode, mode_coe, mode_ord

def extend_coe(GT, maxL_p):
    """
    Extend GT with to maxL_p monomials. Insert 0 for the new values. 

    Used in infer_dynamic_modes_ex_dbs.
    
    """
    # Number of monomials
    L_p = GT.shape[1]

    # Number of variables
    L_y = GT.shape[0]

    NGT = np.zeros((L_y,maxL_p))
    for i in range(0,L_y):
        for j in range(0,L_p):
            NGT[i][-j-1] = GT[i][-j-1]

    return NGT

def infer_dynamic_modes_ex_dbs(t_list, y_list, stepsize, maxorder, ep=0.01):
    """Implementation of clustering performed by DBSCAN.

    Clustering based on coefficient of each mode.

    """
    len_tr = len(y_list)
    # mode = []
    # mode_tls = []
    # mode_yls = []
    # mode_y0l = []
    # mode_tul = []
    # mode_coe = []
    # mode_ord = []
    gene = generate_complete_polynomial(y_list[0].shape[1],maxorder)
    maxL_p = gene.shape[0]
    coefsing = np.zeros((len_tr,y_list[0].shape[1]*maxL_p))
    for l in range(0,len_tr):
        # print("trace"+str(l))
        
        for od in range(0,maxorder):
            od = maxorder
            # print("try of order"+str(od))
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], od, stepsize)
            g = pinv2(A).dot(b)
            t_list_l, y_list_l = simulation_ode(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0, solve_method='BDF')
            dis = dist(y_list[l:l+1],y_list_l)
            cgt = g.T
        
            if dis < ep:
                matarr = extend_coe(cgt,maxL_p).reshape(-1)
                # print(matarr)
                coefsing[l][:] = matarr
                break
        # A, b = diff_method(t_list[l:l+1], y_list[l:l+1], maxorder, stepsize)
        # g = pinv2(A).dot(b)
        # cgt = g.T
        # matarr = cgt.reshape(-1)
        # coefsing[l][:] = matarr
        # print(matarr)


        # Can also consider using KMeans, etc.
        db = skc.DBSCAN(eps=0.1, min_samples=2).fit(coefsing)
        labels = db.labels_
        # num_mode = 3
        # kmeans = skc.KMeans(n_clusters=num_mode, random_state=0)
        # kmeans.fit(coefsing)
        # labels = kmeans.labels_
    return labels

def infer_dynamic_modes_exx(t_list, y_list, stepsize, maxorder, ep=0.01):
    """Similar to infer_dynamic_modes_ex.
    
    Don't try higher orders if lower order is already good enough
    (according to ep).

    """
    len_tr = len(y_list)
    mode = []
    mode_tls = []
    mode_yls = []
    mode_y0l = []
    mode_tul = []
    mode_coe = []
    mode_ord = []
    for l in range(0,len_tr):
        print("trace"+str(l))
        new = 1
        if len(mode) != 0:
            dism = []
            disodm = []
            for i in range(0,len(mode)):
                print("try of mode"+str(i))
                dis = []
                disod = []
                comt = mode_tls[i].copy()
                comy = mode_yls[i].copy()
                comy0 = mode_y0l[i].copy()
                comtu = mode_tul[i].copy()
                comt.append(t_list[l])
                comy.append(y_list[l])
                comy0.append(y_list[l][0])
                comtu.append((t_list[l][0],t_list[l][-1]))
                for od in range(0,maxorder+1):
                    print("try of order"+str(od)+"diff")
                    A, b = diff_method(comt, comy, od, stepsize)
                    g = pinv2(A).dot(b)
                    print("try of order"+str(od)+"simu")
                    comttest, comytest = simulation_ode(ode_test(g.T,od), comy0, comtu, stepsize, eps=0, solve_method='BDF')
                    print("try of order"+str(od)+"dist")
                    dis.append(dist(comy,comytest))
                    disod.append(od)
                p = dis.index(min(dis))
                dism.append(dis[p])
                disodm.append(disod[p])
            if min(dism)<ep:
                new = 0
                q = dism.index(min(dism))
                mode[q].append(l)
                mode_tls[q].append(t_list[l])
                mode_yls[q].append(y_list[l])
                mode_y0l[q].append(y_list[l][0])
                mode_tul[q].append((t_list[l][0],t_list[l][-1]))
                mode_ord[q] = disodm[q]
                A, b = diff_method(mode_tls[q], mode_yls[q], mode_ord[q], stepsize)
                g = pinv2(A).dot(b)
                mode_coe[q] = g.T
        
        if new == 1:
            dis = []
            cgt = []
            for od in range(0,maxorder+1):
                print("new of order"+str(od))
                A, b = diff_method(t_list[l:l+1], y_list[l:l+1], od, stepsize)
                g = pinv2(A).dot(b)
                t_list_l, y_list_l = simulation_ode(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0, solve_method='BDF')
                dis.append(dist(y_list[l:l+1],y_list_l))
                cgt.append(g.T)
            p = dis.index(min(dis))
            mode.append([l])
            mode_tls.append([t_list[l]])
            mode_yls.append([y_list[l]])
            mode_y0l.append([y_list[l][0]])
            mode_tul.append([(t_list[l][0],t_list[l][-1])])
            mode_coe.append(cgt[p])
            mode_ord.append(p)
    
    return mode, mode_coe, mode_ord


def lineg(labell, A_list, b_list):
    """Perform linear regression (try to stack).

    @A_list: matrix A.
    @b_list: vector b.

    """
    l = len(labell)
    print(labell)
    A = A_list[labell[0]]
    b = b_list[labell[0]]
    for i in range(0,l-1):
        np.vstack((A,A_list[labell[i+1]]))
        np.vstack((b,b_list[labell[i+1]]))
    clf = linear_model.LinearRegression(fit_intercept=False)
    # clf=Ridge(alpha=0.1)
    print(A)
    print(b)
    clf.fit(A,b)
    g = clf.coef_
    print("g=",g)
    return g


def distlg(g, label, A_list, b_list):
    """Distance between A * g and b (maximum of absolute value of difference).
    
    """
    
    A = A_list[label]
    b = b_list[label]
    bb = np.matmul(A, g.T)
    dis = bb - b
    maxx = dis.max()
    minn = dis.min()
    dist = max(np.fabs(maxx),np.fabs(minn))
    print("dist = ",dist)
    return dist


def infer_dynamic_modes_pie(t_list, y_list, stepsize, maxorder, ep=0.1):
    """Another clustering method (assume trajectory is already partitioned).

    Works on the coefficients of the ODE.

    """
    len_tr = len(y_list)
    A_list = []
    b_list = []
    label_list=[]
    G = []
    for l in range(0,len_tr):
        print("trace"+str(l))
        A, b = diff_method1(t_list[l:l+1], y_list[l:l+1], maxorder, stepsize)
        A_list.append(A)
        b_list.append(b)
        label_list.append(l)
    print(label_list)
    labell = []
    while len(label_list)>0:
        label = [label_list[0]]
        print(label_list[0])
        g = lineg([label_list[0]],A_list,b_list)
        for l in range(0,len(label_list)):
            if distlg(g,label_list[l],A_list,b_list) < ep:
                label.append(label_list[l])
        for l in range(0,len(label)):
            if label[l] in label_list:
                label_list.remove(label[l])
        print(label)
        g = lineg(label,A_list,b_list)
        G.append(g)
        labell.append(label)

    return G, labell

def diff_method_new(t_list, y_list, order, stepsize):
    """Using multi-step difference method (BDF) to calculate the
    coefficient matrix.

    """
    final_A_mat = None
    final_b_mat = None
    final_y_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)
    L_p = gene.shape[0]

    for k in range(0,len(y_list)):
        t_points = t_list[k]
        L_t = len(t_points)
        D = L_t - 5    #order5
        y_points = y_list[k]
        A_matrix = np.zeros((D, L_p), dtype=np.double)
        b_matrix = np.zeros((D, L_y),  dtype=np.double)
        y_matrix = np.zeros((D, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        for i in range(0, D):
                # forward
            A_matrix[i] = coef_matrix[i+5]
            b_matrix[i] = (137 * y_points[i+5] - 300 * y_points[i+4] + 300 * y_points[i+3] -
                           200 * y_points[i+2] + 75 * y_points[i+1] - 12 * y_points[i]) / (60 * stepsize)
            y_matrix[i] = y_points[i+5]
        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
            final_y_mat = y_matrix

        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]
            final_y_mat = np.r_[final_y_mat,y_matrix]

    return final_A_mat, final_b_mat, final_y_mat


def diff_method_backandfor(t_list, y_list, order, stepsize):
    """Using multi-step difference method (BDF) to calculate the
    coefficient matrix.

    """
    final_A_mat = None
    final_b1_mat = None
    final_b2_mat = None
    final_y_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)
    L_p = gene.shape[0]

    for k in range(0,len(y_list)):
        t_points = t_list[k]
        L_t = len(t_points)
        D = L_t - 5    #order5
        y_points = y_list[k]
        A_matrix = np.zeros((D-5, L_p), dtype=np.double)
        b1_matrix = np.zeros((D-5, L_y),  dtype=np.double)
        b2_matrix = np.zeros((D-5, L_y),  dtype=np.double)
        y_matrix = np.zeros((D-5, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        for i in range(5, D):
                # forward
            A_matrix[i-5] = coef_matrix[i]
            b1_matrix[i-5] = (137 * y_points[i] - 300 * y_points[i-1] + 300 * y_points[i-2] -
                           200 * y_points[i-3] + 75 * y_points[i-4] - 12 * y_points[i-5]) / (60 * stepsize)
            b2_matrix[i-5] = (-137 * y_points[i] + 300 * y_points[i+1] - 300 * y_points[i+2] + 
                            200 * y_points[i+3] - 75 * y_points[i+4] + 12 * y_points[i+5])/ (60 * stepsize)
            y_matrix[i-5] = y_points[i]
        if k == 0:
            final_A_mat = A_matrix
            final_b1_mat = b1_matrix
            final_b2_mat = b2_matrix
            final_y_mat = y_matrix

        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b1_mat = np.r_[final_b1_mat,b1_matrix]
            final_b2_mat = np.r_[final_b2_mat,b2_matrix]
            final_y_mat = np.r_[final_y_mat,y_matrix]

    return final_A_mat, final_b1_mat, final_b2_mat, final_y_mat


def diff_method_new1(t_list, y_list, order, stepsize):
    """Using multi-step difference method (Adams) to calculate the
    coefficient matrix.

    """
    final_A_mat = None
    final_b_mat = None
    final_y_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)
    L_p = gene.shape[0]

    for k in range(0,len(y_list)):
        t_points = t_list[k]
        L_t = len(t_points)
        D = L_t - 4    #order5
        y_points = y_list[k]
        A_matrix = np.zeros((D, L_p), dtype=np.double)
        b_matrix = np.zeros((D, L_y),  dtype=np.double)
        y_matrix = np.zeros((D, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        for i in range(0, D):
                # forward
            A_matrix[i] = (-19*coef_matrix[i]+106*coef_matrix[i+1]-264*coef_matrix[i+2]+646*coef_matrix[i+3]+251*coef_matrix[i+4])*stepsize
            b_matrix[i] = (y_points[i+4]-y_points[i+3])*720
            y_matrix[i] = y_points[i+4]
        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
            final_y_mat = y_matrix

        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]
            final_y_mat = np.r_[final_y_mat,y_matrix]

    return final_A_mat, final_b_mat, final_y_mat

def compare(A,B,ep):
    """Check the relative difference |A - B| / A is larger/smaller
    than ep.
    
    """
    C = A - B
    r = 1
    # for i in range(0,C.shape[0]):
    #     for j in range(0,C.shape[1]):
    #         c = abs(C[i,j])
    #         a = abs(A[i,j])*ep
    #         # a = ep
    #         if c >= a:
    #             r = 0
    for i in range(0,C.shape[0]):
        a = 0
        b = 0
        c = 0
        for j in range(0,C.shape[1]):
            c += C[i,j]**2
            a += A[i,j]**2
            b += B[i,j]**2
        f1 = np.sqrt(c)
        f2 = np.sqrt(a) + np.sqrt(b)
        if f1 > ep*f2:
            r = 0
    return r

def matrowex(matr, l):
    """Pick some rows of a matrix to form a new matrix."""
    finalmat = None
    for i in range(0,len(l)):
        if i == 0:
            finalmat = np.mat(matr[l[i]])
        else:
            finalmat = np.r_[finalmat,np.mat(matr[l[i]])]
    return finalmat

def infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1):
    """Implementation of Alur's method (extended to nonlinear case).

    """
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    leng = Y.shape[0]
    label_list = []
    drop = []
    for i in range(0,leng):
        label_list.append(i)

    P = []
    G = []

    while len(label_list)>0:
        print("label",len(label_list))
        p = random.choice(label_list)
        clf = linear_model.LinearRegression(fit_intercept=False)
        # print("fitstart")
        # print("fitend")
        if p > 0 and p < leng - 1:
            pp = [p-1,p,p+1]
        elif p == 0:
            pp = [p,p+1,p+2]
        else:
            pp = [p-2,p-1,p]
        clf.fit (matrowex(A,pp), matrowex(b,pp))
        pre = clf.predict(matrowex(A,pp))
        ppp = pp[:]
        # print("first")
        retation = 0
        while compare(matrowex(b,pp),pre,ep) ==1:
            retation += 1
            # print("compare",retation,len(pp))
            ppp = pp[:]
            reta = 0
            # if pp[0]>0 and pp[0]-1 in label_list:
            if pp[0]>0 :  
                pre1 = clf.predict(np.mat(A[pp[0]-1]))
                if compare(np.mat(b[pp[0]-1]),pre1,ep) ==1 :
                    pp.insert(0,pp[0]-1)
                    reta = 1
            # if pp[-1]<leng-1 and pp[-1]+1 in label_list:
            if pp[-1]<leng-1:   
                pre2 = clf.predict(np.mat(A[pp[-1]+1]))
                if compare(np.mat(b[pp[-1]+1]),pre2,ep) ==1 :
                    pp.append(pp[-1]+1)
                    reta = 1
            
            if reta == 0 or retation >= 20:
                break

            clf.fit (matrowex(A,pp), matrowex(b,pp))
            pre = clf.predict(matrowex(A,pp))
        
        if len(ppp)<10:
            label_list.remove(p)
            drop.append(p)
            continue
        
        pp = ppp[1:]
        print("second")
        while len(ppp) > len(pp):
            print("ppp",len(ppp))
            pp = ppp[:]
            ppp = []
            clf.fit (matrowex(A,pp), matrowex(b,pp))
            for i in range(0,len(label_list)):
                prew = clf.predict(np.mat(A[label_list[i]]))
                if compare(np.mat(b[label_list[i]]),prew,ep) ==1 :
                    ppp.append(label_list[i])
        
        gp = ppp[:]
        if len(gp) < 10:
            label_list.remove(p)
        else:
            g = clf.coef_
            P.append(gp)
            G.append(g)
            for ele in gp:
                label_list.remove(ele)

    drop.sort() 

    return P,G,drop


def reclass(A,b,P,ep):
    lenofp = [(len(P[i]), i) for i in range(len(P))]
    lenofp = sorted(lenofp,reverse=True)
    sortl = [i for _, i in lenofp]
    sortP = []
    for i in sortl:
        sortP.append(P[i])
    
    # print(sortP)
    newP = []
    while len(sortP) > 1:
        ssortP = sortP[:]
        l = [0]
        for i in range(1,len(sortP)):
            pp = ssortP[0]+ssortP[i]
            # print(pp)
            clf = linear_model.LinearRegression(fit_intercept=False)
            clf.fit (matrowex(A,pp), matrowex(b,pp))
            pre = clf.predict(matrowex(A,pp))
            if compare(matrowex(b,pp),pre,ep) ==1 :
                l.append(i)
        
        pp = []
        for j in l:
            pp.extend(ssortP[j])
            sortP.remove(ssortP[j])
        
        newP.append(pp)
    
    if len(sortP) == 1:
        newP.append(sortP[0])

    G = []
    for i in range(len(newP)):
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit (matrowex(A,newP[i]), matrowex(b,newP[i]))
        g = clf.coef_
        G.append(g)



    return newP,G
            


def dropclass(P,G,D,A,b,Y,ep,stepsize):
    DD = D[:]
    for i in DD:
        if i<(Y.shape[0] - 5):
            co = np.mat(A[i])
            forw = np.mat(b[i])
            # print(forw.shape)
            back = (-137 * Y[i] + 300 * Y[i+1] - 300 * Y[i+2] + 
                            200 * Y[i+3] - 75 * Y[i+4] + 12 * Y[i+5])/ (60 * stepsize)
            back = np.mat(back)
            # print(back.shape)
            for j in range(0,len(G)):
                g = G[j]
                der = np.matmul(co,g.T)
                # print(der.shape)
                if compare(forw,der,ep) or compare(back,der,ep):
                    P[j].append(i)
                    D.remove(i)
                    break

    return P,D
            
                


        









def infer_multi_linear(t_list, y_list, stepsize, maxorder, ep=0.001):
    num_neigh = 30
    num_step = 50
    num_mode = 2

    # First apply Linear Multistep Method
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    num_pt = Y.shape[0]

    # Get neighborhood of a point
    def get_neigh(p):
        dists = [((A[p]-A[i]).dot(A[p]-A[i]), i) for i in range(num_pt)]
        dists = sorted(dists)
        return [i for _, i in dists[:num_neigh]]

    # Predict for a given point
    def predict_for_pt(p):
        clf = linear_model.LinearRegression(fit_intercept=False)
        
        neigh = get_neigh(p)
        clf.fit(matrowex(A, neigh), matrowex(b, neigh))
        diff_mat = clf.predict(matrowex(A, neigh)) - matrowex(b, neigh)
        return np.square(diff_mat).sum(), clf.coef_

    # Apply KMeans
    res = []
    chosen_pts = list(np.random.permutation(num_pt)[:num_step])
    for p in chosen_pts:
        err, coef = predict_for_pt(p)
        if err < ep:
            res.append((err, p, coef))

    num_coeff = res[0][2].shape[0] * res[0][2].shape[1]
    cluster_res = [res[i][2].reshape((num_coeff,)) for i in range(len(res))]
    kmeans = skc.KMeans(n_clusters=num_mode, random_state=0)
    kmeans.fit(cluster_res)

    # Collect point for each mode and fit again
    mode_pts = []
    for i in range(num_mode):
        mode_pts.append([])
    for i, lab in enumerate(kmeans.labels_):
        mode_pts[lab].extend(get_neigh(res[i][1]))
    for i in range(num_mode):
        mode_pts[i] = sorted(list(set(mode_pts[i])))

    # Get final results
    clfs = []
    for i in range(num_mode):
        clf = linear_model.LinearRegression(fit_intercept=False)

        clf.fit(matrowex(A, mode_pts[i]), matrowex(b, mode_pts[i]))
        clfs.append(clf)

    # Perform classification using SVM
    def predict_point(p):
        diffs = []
        for i in range(num_mode):
            pred_b = clfs[i].predict(matrowex(A, [p]))
            actual_b = matrowex(b, [p])
            diff = np.square(pred_b - actual_b).sum()
            diffs.append((diff, i))
        return sorted(diffs)[0][1]

    labels = []
    positions = []
    for i in range(num_pt):
        if predict_point(i) == 0:
            labels.append(-1)
        else:
            labels.append(1)
        positions.append({1: Y[i,0], 2: Y[i,1], 3: Y[i,2]})

    prob = svm_problem(labels, positions)
    param = svm_parameter('-t 1 -d 1 -c 10 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    p_label, p_acc, p_val = svm_predict(labels, positions, m)

    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()

    g = -m.rho[0]
    a1 = 0
    a2 = 0
    a3 = 0
    for i in range(0,nsv):
        a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
        a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
        a2 = a2 + svc[i][0] * 0.5 * sv[i][3]

    print("a1",a1)
    print("a2",a2)
    print("a3",a3)
    print("g",g)

    return clfs, [a1, a2, a3, g]




def infer_multi_linear_new(t_list, y_list, stepsize, maxorder, ep=0.001):
    num_neigh = 30
    num_step = 50
    num_mode = 2

    # First apply Linear Multistep Method
    A, b1, b2, Y = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    num_pt = Y.shape[0]
    num_dim = Y.shape[1]
    
    def valid(p):
        if norm(b1[p]-b2[p])/(norm(b1[p])+norm(b2[p])) < 0.02:
            return True
        else:
            return False

    
    # Get neighborhood of a point
    def get_neigh(p):
        dists = [((A[p]-A[i]).dot(A[p]-A[i]), i) for i in range(num_pt) if valid(i)]
        dists = sorted(dists)
        return [i for _, i in dists[:num_neigh]]

    # Predict for a given point
    def predict_for_pt(p):
        clf = linear_model.LinearRegression(fit_intercept=False)
        
        neigh = get_neigh(p)
        clf.fit(matrowex(A, neigh), matrowex(b1, neigh))
        diff_mat = clf.predict(matrowex(A, neigh)) - matrowex(b1, neigh)
        return np.square(diff_mat).sum(), clf.coef_

    # Apply KMeans
    res = []
    chosen_pts = list(np.random.permutation(num_pt)[:num_step])
    for p in chosen_pts:
        if valid(p):
            err, coef = predict_for_pt(p)
            if err < ep:
                res.append((err, p, coef))

    num_coeff = res[0][2].shape[0] * res[0][2].shape[1]
    cluster_res = [res[i][2].reshape((num_coeff,)) for i in range(len(res))]
    kmeans = skc.KMeans(n_clusters=num_mode, random_state=0)
    kmeans.fit(cluster_res)

    # Collect point for each mode and fit again
    mode_pts = []
    for i in range(num_mode):
        mode_pts.append([])
    for i, lab in enumerate(kmeans.labels_):
        mode_pts[lab].extend(get_neigh(res[i][1]))
    for i in range(num_mode):
        mode_pts[i] = sorted(list(set(mode_pts[i])))

    # Get final results
    clfs = []
    for i in range(num_mode):
        clf = linear_model.LinearRegression(fit_intercept=False)

        clf.fit(matrowex(A, mode_pts[i]), matrowex(b1, mode_pts[i]))
        clfs.append(clf)

    # Perform classification using SVM
    def predict_point(p):
        diffs = []
        for i in range(num_mode):
            pred_b = clfs[i].predict(matrowex(A, [p]))
            actual_b = matrowex(b1, [p])
            diff = np.square(pred_b - actual_b).sum()
            diffs.append((diff, i))
        return sorted(diffs)[0][1]

    labels = []
    positions = []
    for i in range(num_pt):
        if valid(i):
            if predict_point(i) == 0:
                labels.append(-1)
            else:
                labels.append(1)
            dic = dict()
            for j in range(num_dim): 
                dic[j+1] = Y[i,j]

            positions.append(dic)

    prob = svm_problem(labels, positions)
    param = svm_parameter('-t 1 -d 1 -c 10 -r 1 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    p_label, p_acc, p_val = svm_predict(labels, positions, m)

    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print('svc',svc)
    # print('sv',sv)

    g = -m.rho[0]
    # a1 = 0
    # a2 = 0
    # a3 = 0
    a = np.zeros(num_dim)
    # print('a',a)
    for j in range(num_dim):
        for i in range(0,nsv):
            # a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
            # a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
            # a2 = a2 + svc[i][0] * 0.5 * sv[i][3]
            a[j] = a[j] + svc[i][0] * 0.5 * sv[i][j+1]
    
    for i in range(0,nsv):
        g = g + svc[i][0]*1

    # print("a1",a1)
    # print("a2",a2)
    # print("a3",a3)
    # print("g",g)

    return clfs, [a, g]


def test_classify(f, clfs, boundary, maxorder, x):
    """Test a classification."""
    # print('bound',boundary)
    a, g = boundary

    def classify_mode(x):
        # return a1 * x[0] + a2 * x[1] + a3 * x[2] + g > 0
        return a.dot(x.T) + g >0

    def get_poly_pt(x):
        gene = generate_complete_polynomial(len(x), maxorder)
        val = []
        for i in range(gene.shape[0]):
            val.append(1.0)
            for j in range(gene.shape[1]):
                val[i] = val[i] * (x[j] ** gene[i,j])
        return val

    def predict_deriv(x):
        poly_pt = np.mat(get_poly_pt(x))
        if classify_mode(x):
            return clfs[1].predict(poly_pt)
        else:
            return clfs[0].predict(poly_pt)

    diff1 = np.array(f(0.0, x))
    # print('diff1',diff1)
    # print('nd1',norm(diff1))
    diff2 = predict_deriv(x).ravel()
    # print('diff2',diff2)
    # print('nd2',norm(diff2))
    err = norm(diff1-diff2)/(norm(diff1)+norm(diff2))
    return err

 


    
        




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
    
