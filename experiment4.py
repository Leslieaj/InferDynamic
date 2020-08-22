import numpy as np
from fractions import Fraction
from generator import generate_complete_polynomial
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
# from dynamics import dydx3, fvdp2_1, fvdp3_1, fvdp3_2, fvdp3_3, \
#     mode2_1, mode2_11, mode2_1_test, \
#     conti_test, conti_test_test, conti_test1, ode_test, \
#     mode1, mode2, event2, \
#     mmode1, mmode2, event3, mmode, \
#     incubator_mode1, incubator_mode2, event1, \
#     modetr_1, modetr_2, modetr_3, eventtr_1, eventtr_2
from infer_multi_ch import simulation_ode, infer_dynamic, parti, infer_dynamic_modes_ex, norm, reclass, dropclass, \
    infer_dynamic_modes_exx, dist, diff_method, diff_method1, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie, \
    infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor, \
        infer_model, test_model, segment_and_fit, merge_cluster2, matrowex, merge_cluster_tol, rel_diff, merge_cluster_tol2,\
            diff

import infer_multi_ch
from generator import generate_complete_polynomial
import dynamics
import warnings
warnings.filterwarnings('ignore')

import math
import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *



mmode_params = [
    [1,5,5,0.1],
]

def get_mmode(param_id):
    s,r1,r2,p = mmode_params[param_id]

    def mmode1(t,y):
        y0, y1, y2, y3 = y
        dydt = [-y1*s,y0-r1,y3,-p*(y2+r2)**2]
        return dydt
    

    def mmode2(t,y):
        y0, y1, y2, y3 = y
        dydt = [-y1*s,y0-r1,y3,p*(y0 - y2)**2]
        return dydt

    return [mmode1,mmode2]

mmode = get_mmode(0)


def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator


def get_event(param_id):
    @eventAttr()
    def event1(t,y):
        y0, y1, y2, y3 = y
        return y2
    return event1

event1 = get_event(0)


cases = {
    0: {
        'params': 0,
        'y0': [[4,0.1,3.1,0], [5.9,0.2,-3,0], [4.1,0.5,2,0], [6,0.7,2,0]],
        'y0_test': [[4.6,0.13,2,0], [5.3,0.17,-2,0]],
        't_tuple': 5,
        'stepsize': 0.01,
        'ep': 0.01,
        'mergeep': 0.01
    },
    1: {
        'params': 0,
        'y0': [[4.1,0.5,2,0], [6,0.7,2,0]],
        'y0_test': [[4.6,0.13,2,0], [5.3,0.17,-2,0]],
        't_tuple': 5,
        'stepsize': 0.01,
        'ep': 0.01,
        'mergeep': 0.01
    },
    2: {
        'params': 0,
        'y0': [[4,0.1,3.1,0], [5.9,0.2,-3,0], [4.1,0.5,2,0], [6,0.7,2,0]],
        'y0_test': [[4.6,0.13,2,0], [5.3,0.17,-2,0]],
        't_tuple': 5,
        'stepsize': 0.002,
        'ep': 0.01,
        'mergeep': 0.01
        
    },
    3: {
        'params': 0,
        'y0': [[4,0.1,3.1,0], [5.9,0.2,-3,0], [6,0.7,2,0]],
        'y0_test': [[4.6,0.13,2,0], [5.3,0.17,-2,0]],
        't_tuple': 10,
        'stepsize': 0.01,
        'ep': 0.01,
        'mergeep': 0.005
    },
}

def case(y0,t_tuple,stepsize,maxorder,modelist,event,ep,method):
    t_list, y_list = simulation_ode_2(modelist, event, y0, t_tuple, stepsize)
    
    
    if method == "new":
        
        
        A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
        P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
        P,D=dropclass(P,G,D,A,b,Y,0.01,stepsize)
        # print(len(P))

        y = []
        x = []

        for j in range(0,len(P[0])):
            y.append(1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1], 3:Y[P[0][j],2], 4:Y[P[0][j],3]})
        
        for j in range(0,len(P[1])):
            y.append(-1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1], 3:Y[P[1][j],2], 4:Y[P[1][j],3]})

        prob  = svm_problem(y, x)
        param = svm_parameter('-t 1 -d 1 -c 10 -r 1 -b 0 -q')
        m = svm_train(prob, param)
        svm_save_model('model_file', m)
        nsv = m.get_nr_sv()
        svc = m.get_sv_coef()
        sv = m.get_SV()
        g = -m.rho[0]
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
        for i in range(0,nsv):
            a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
            a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
            a3 = a3 + svc[i][0] * 0.5 * sv[i][3]
            a4 = a4 + svc[i][0] * 0.5 * sv[i][4]
            g = g + svc[i][0]
        def f(x):
            return a1*x[0] + a2*x[1] + a3*x[2] + a4*x[3] + g > 0
        
    sum = 0
    num = 0
    
    def get_poly_pt(x):
        gene = generate_complete_polynomial(len(x), maxorder)
        val = []
        for i in range(gene.shape[0]):
            val.append(1.0)
            for j in range(gene.shape[1]):
                val[i] = val[i] * (x[j] ** gene[i,j])
        poly_pt = np.mat(val)
        return poly_pt
    
    for ypoints in y_list:
        num = num + ypoints.shape[0]
        for i in range(ypoints.shape[0]):
            if event(0,ypoints[i])>0:
                exact = modelist[0](0,ypoints[i])
            else:
                exact = modelist[1](0,ypoints[i])
            if f(ypoints[i]) == 1:
                predict = np.matmul(get_poly_pt(ypoints[i]),G[0].T)
            else:
                predict = np.matmul(get_poly_pt(ypoints[i]),G[1].T)

            exact = np.mat(exact)
            diff = exact - predict
            c=0
            a=0
            b=0
            for j in range(diff.shape[1]):
                c=c+diff[0,j]**2
                a=a+exact[0,j]**2
                b=b+predict[0,j]**2
            f1 = np.sqrt(c)
            f2 = np.sqrt(a)+np.sqrt(b)
            sum = sum + f1/f2

    return sum/num

def case1():
    y0 = [[4,0.1,3.1,0], [5.9,0.2,-3,0], [4.1,0.5,2,0], [6,0.7,2,0]]
    y0_test = [[4.6,0.13,2,0], [5.3,0.17,-2,0]]
    T = 5
    stepsize = 0.01
    ep = 0.01
    maxorder = 2
    boundary_order = 1
    num_mode = 2
    method = 'tolmerge'
    t_list, y_list = simulation_ode_2(mmode, event1, y0, T, stepsize)
    t_test_list, y_test_list = simulation_ode_2(mmode, event1, y0_test, T, stepsize)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    np.savetxt("A4.txt",A,fmt='%8f')
    np.savetxt("b4.txt",b,fmt='%8f')
    YT, FT = diff(t_list+t_test_list, y_list+y_test_list, dynamics.modeex4)
    np.savetxt("YT4.txt",YT,fmt='%8f')
    np.savetxt("FT4.txt",FT,fmt='%8f')
    # P, G, boundary = infer_model(
    #             t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
    #             num_mode=num_mode, modelist=mmode, event=event1, ep=ep, method=method, verbose=False)
    
    # d_avg = test_model(
    #             P, G, boundary, num_mode, y_list, mmode, event1, maxorder, boundary_order)
    # print(d_avg)
    # print(G)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # f = open("out.txt", "w")
    # for i in range(0,Y.shape[0]):
    #     if event1(0,Y[i])>0:
    #         print(1,':',rel_diff(b1[i],b2[i]),b1[i],b2[i],mmode[0](0,Y[i]),file = f)
    #     else:
    #         print(-1,':',rel_diff(b1[i],b2[i]),b1[i],b2[i],mmode[1](0,Y[i]),file = f)
    # f.close()

    # res, drop, clfs = segment_and_fit(A, b1, b2, ytuple)
    # bt = np.zeros((b1.shape[0],b1.shape[1]))
    # for i in range(0,A.shape[0]):
    #     if event1(0,Y[i]) > 0:
    #         bt[i] = mmode[0](0,Y[i])
    #     else:
    #         bt[i] = mmode[1](0,Y[i])

if __name__ == "__main__":
    case1()