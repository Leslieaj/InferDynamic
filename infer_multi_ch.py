import numpy as np
from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
import math
import random

from dynamics import dydx1, dydx2, fvdp2, fvdp2_1, ode_test
from generator import generate_complete_polynomail
from draw import draw, draw2D

import sklearn.cluster as skc
from sklearn import metrics   # 评估模型
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model
from sklearn.linear_model import Ridge

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
        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        y_object = solve_ivp(ode_func, (t_start, t_end + 1.1*stepsize), y0[k], t_eval = t_points, method ='RK45', rtol=1e-7, atol=1e-9, max_step = stepsize/100)
        y_points = y_object.y.T
        if eps > 0:
            for i in range(0,y_points.shape[0]):
                for j in range(0,y_points.shape[1]):
                    y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
        t_list.append(t_points)
        y_list.append(y_points)
    return t_list, y_list


def simulation_ode_stiff(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0):
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
        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        y_object = solve_ivp(ode_func, (t_start, t_end + 1.1*stepsize), y0[k], t_eval = t_points, method ='BDF', rtol=1e-7, atol=1e-9)
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


def diff_method1(t_list, y_list, order, stepsize):
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
        D = L_t - 5    #bdf5
        y_points = y_list[k]
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
        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]

    return final_A_mat, final_b_mat

def extend_coe(GT, maxL_p):
    L_p = GT.shape[1]
    L_y = GT.shape[0]
    NGT = np.zeros((L_y,maxL_p))
    # print(L_p)
    # print(L_y)
    # print(maxL_p)
    # print(GT)
    # print(NGT)
    for i in range(0,L_y):
        for j in range(0,L_p):
            NGT[i][-j-1] = GT[i][-j-1]
    # print(NGT)
    return NGT


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
        parpo = [-1]
        diffmat2 = np.multiply(diffmat,diffmat)
        diffvecm = diffmat2.sum(axis=1)
        for i in range(0,row-2):
            parbool = 0
            if diffvecm[i+1]/diffvecm[i] > 1+ep or diffvecm[i+1]/diffvecm[i] < 1-ep:
                parbool = 1
            if np.dot(diffmat[i+1],diffmat[i].T)/math.sqrt(diffvecm[i+1]*diffvecm[i]) < np.sin(an*np.pi):
                parbool = 1
            if parbool == 1 and parpo[-1] != i:
                parpo.append(i+1)
        parpo.append(row-1)
        for i in range(0,len(parpo)-1):
            tpar_list.append(t_points[parpo[i]+1:parpo[i+1]])
            ypar_list.append(y_points[parpo[i]+1:parpo[i+1]][:])
    return tpar_list, ypar_list


def dist(y_list,y_list_test):
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
                comttest, comytest = simulation_ode_stiff(ode_test(g.T,od), comy0, comtu, stepsize, eps=0)
                dis.append(dist(comy,comytest))
                cgt.append(g.T)
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], od, stepsize)
            g = pinv2(A).dot(b)
            t_list_l, y_list_l = simulation_ode_stiff(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
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

def infer_dynamic_modes_ex_dbs(t_list, y_list, stepsize, maxorder, ep=0.01):
    len_tr = len(y_list)
    mode = []
    mode_tls = []
    mode_yls = []
    mode_y0l = []
    mode_tul = []
    mode_coe = []
    mode_ord = []
    gene = generate_complete_polynomail(y_list[0].shape[1],maxorder)
    maxL_p = gene.shape[0]
    coefsing = np.zeros((len_tr,y_list[0].shape[1]*maxL_p))
    for l in range(0,len_tr):
        print("trace"+str(l))
        
        for od in range(0,maxorder):
            print("try of order"+str(od))
            A, b = diff_method(t_list[l:l+1], y_list[l:l+1], od, stepsize)
            g = pinv2(A).dot(b)
            t_list_l, y_list_l = simulation_ode_stiff(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
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


        db = skc.DBSCAN(eps=0.1, min_samples=2).fit(coefsing)
        labels = db.labels_
    return labels

def infer_dynamic_modes_exx(t_list, y_list, stepsize, maxorder, ep=0.01):
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
                    comttest, comytest = simulation_ode_stiff(ode_test(g.T,od), comy0, comtu, stepsize, eps=0)
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
                t_list_l, y_list_l = simulation_ode_stiff(ode_test(g.T,od), [y_list[l][0]], [(t_list[l][0],t_list[l][-1])], stepsize, eps=0)
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



def distlg(g,label,A_list, b_list):
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


        
    
    return G,labell

def diff_method_new(t_list, y_list, order, stepsize):
    """Using multi-step difference method (bdf) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    final_y_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomail(L_y,order)
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
            A_matrix[i] = 60 * stepsize * coef_matrix[i+5]
            b_matrix[i] = 137 * y_points[i+5] - 300 * y_points[i+4] + 300 * y_points[i+3] - 200 * y_points[i+2] + 75 * y_points[i+1] - 12 * y_points[i]
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


def diff_method_new1(t_list, y_list, order, stepsize):
    """Using multi-step difference method (bdf) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    final_y_mat = None
    
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomail(L_y,order)
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
    C = A - B
    r = 1
    for i in range(0,C.shape[0]):
        for j in range(0,C.shape[1]):
            c = abs(C[i,j])
            a = abs(A[i,j])*ep
            # a = ep
            if c >= a:
                r = 0
    return r

def matrowex(matr,l):
    finalmat = None
    for i in range(0,len(l)):
        if i == 0:
            finalmat = np.mat(matr[l[i]])
        else:
            finalmat = np.r_[finalmat,np.mat(matr[l[i]])]
    return finalmat

def infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1):

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
        print("fitstart")
        clf.fit (np.mat(A[p]), np.mat(b[p]))
        g = clf.coef_
        print("fitend")
        pp = [p]
        pre = clf.predict(matrowex(A,pp))
        print("first")
        retation = 0
        while compare(matrowex(b,pp),pre,ep) ==1:
            retation += 1
            print("compare",retation,len(pp))
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
            
            if reta == 0 or retation >= 10:
                break

            clf.fit (matrowex(A,pp), matrowex(b,pp))
            pre = clf.predict(matrowex(A,pp))
        
        if len(ppp)<3:
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
        g = clf.coef_
        P.append(gp)
        G.append(g)
        for ele in gp:
            label_list.remove(ele)

    drop.sort() 

    return P,G,drop


        



def test_infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1):
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1)
    maxdis = 0
    maxredis = 0
    for i in range(0,len(P)):
        P[i].sort()
        for j in range(0,len(P[i])):
            if P[i][j+1] - P[i][j] == 1:
                y0 = [Y[P[i][j]]]
                t_tuple = [(0,stepsize)]
                ttestlist, ytestlist = simulation_ode(ode_test(G[i],order), y0, t_tuple, stepsize/100, eps=0)
                disvec = np.mat(ytestlist[0][-1]) - np.mat(Y[P[i][j+1]])
                for k in range(0,disvec.shape[1]):
                    maxdis = max(maxdis,abs(disvec[0,k]))
                    maxredis = max(maxredis,abs(disvec[0,k]/Y[P[i][j+1],k]))
    
    return maxdis,maxredis


 


    
        




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
    
