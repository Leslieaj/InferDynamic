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
from dynamics import ode_test
from infer_multi_ch import simulation_ode, infer_dynamic, parti, infer_dynamic_modes_ex, norm, reclass, dropclass, \
        infer_dynamic_modes_exx, dist, diff_method, diff_method1, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie, \
        infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor, \
        infer_model, test_model, segment_and_fit, kmeans_cluster, dbscan_cluster, diff, merge_cluster_tol2, matrowex, disc

# from infer_multi_ch import infer_model, test_model, simulation_ode_2, simulation_ode_3
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



mode2_params = [
    [-0.026, -1, -0.026, 1, 98.5],
    [-0.26, -1, -0.26, 1, 0],
]

def get_mode2(param_id):
    a1, b1, a2, b2, _ = mode2_params[param_id]

    def mode2_1(t,y):
        """ A hybrid automaton with 2 modes for an incubator.

        Returns derivative at point y = (y0, y1).

        """
        y0, y1 = y
        dydt = [a1 * (y0-y1), b1]
        return dydt
    

    def mode2_2(t,y):
        y0, y1 = y
        dydt = [a2 * (y0-y1), b2]
        return dydt
    return [mode2_1,mode2_2]

mode2 = get_mode2(0)

def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator


def get_event1(param_id):
    _, _, _, _, c0 = mode2_params[param_id]

    @eventAttr()
    def event1(t,y):
        y0, y1 = y
        return y0 - c0

    return event1

event1 = get_event1(0)

cases = {
    0: {
        'params': 0,
        'y0': [[99.5, 80], [97.5, 100]],
        'y0_test': [[100.5, 90], [96, 80]],
        't_tuple': 50,
        'stepsize': 0.1,
    },
    1: {
        'params': 0,
        'y0': [[99.5, 80], [97.5, 100], [100.5, 90]],
        'y0_test': [[96, 80], [99.5, 100]],
        't_tuple': 50,
        'stepsize': 0.1,
    },
    2: {
        'params': 0,
        'y0': [[99.5, 80], [97.5,100]],
        'y0_test': [[100.5, 90], [96, 80]],
        't_tuple': 50,
        'stepsize': 0.5,
    },
    3: {
        'params': 0,
        'y0': [[99.5, 80], [97.5, 100], [100.5, 90]],
        'y0_test': [[96, 80], [99.5, 100]],
        't_tuple': 50,
        'stepsize': 0.5,
    },
    4: {
        'params': 1,
        'y0': [[1, 3], [-1, -2]],
        'y0_test': [[1, 2], [-1, 1]],
        't_tuple': 10,
        'stepsize': 0.1,
    }
}


def case(y0, t_tuple, stepsize, maxorder, modelist, event, ep, method):
    # print('Simulating')
    t_list, y_list = simulation_ode_2(modelist, event, y0, t_tuple, stepsize)
    draw2D(y_list)

    if method == "new":
        # print('Classifying')
        A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
        P, G, D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep)
        P, G = reclass(A, b, P, ep)
        print(G)
        P, D = dropclass(P, G, D, A, b, Y, ep, stepsize)
        # print('Number of modes:', len(P))

        y = []
        x = []

        for j in range(0,len(P[0])):
            y.append(1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})
        
        for j in range(0,len(P[1])):
            y.append(-1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})

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
        for i in range(0,nsv):
            a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
            a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
            g = g + svc[i][0]
        def f(x):
            return a1*x[0] + a2*x[1] + g > 0
        
        print(a1/a1,a2/a1,g/a1 )
        
    sum = 0
    num = 0

    @eventAttr()
    def eventtest(t,y):
        y0, y1 = y
        return a1 * y0 + a2 * y1 + g

    ttest_list, ytest_list = simulation_ode_2([ode_test(G[0],maxorder),ode_test(G[1],maxorder)], eventtest, y0, t_tuple, stepsize)
    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='b',label='Original')
        else:
            plt.plot(y0_list,y1_list,c='b')
    for i, temp_y in enumerate(ytest_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='r', label='Inferred')
        else:
            plt.plot(y0_list,y1_list,c='r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
    
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
    mode2 = get_mode2(0)
    event1 = get_event1(0)
    y0 = [[70,70]]
    stepsize = 1
    maxorder = 1
    boundary_order = 1
    num_mode = 2
    T = 200
    ep = 0.01
    method='kmeans'
    t_list, y_list = simulation_ode_2(mode2, event1, y0, T, stepsize)
    # P,G,C = infer_model(
    #              t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
    #              num_mode=num_mode, modelist=mode2, event=event1, ep=ep, method=method, verbose=False)
    # A, b1, b2, Y = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # num_pt = Y.shape[0]

    # Segment and fit
    # P, drop, clfs = segment_and_fit(A, b1, b2)
    # P, G = dbscan_cluster(clfs, P, A, b1, num_mode)
    # P, _ = dropclass(P, G, drop, A, b1, Y, ep, stepsize)
    # for i in range(0,len(P)):
    #     y0_list = []
    #     y1_list = []
    #     for j in range(0,len(P[i])):
    #         y0_list.append(Y[P[i][j],0])
    #         y1_list.append(Y[P[i][j],1])
    
    #     plt.scatter(y0_list,y1_list,s=1)
    # plt.show()
    # for i, temp_y in enumerate(y_list):
    #     y0_list = temp_y.T[0]
    #     # y0_list = t_list[i]
    #     y1_list = temp_y.T[1]
    #     plt.scatter(y0_list,y1_list,c='b',s=5)
    #     plt.scatter(y0_list[59],y1_list[59],c='r',s=5)

    #     # plt.scatter(y0_list,y1_list,c='b',s=5, label='x2')
    #     # y1_list = temp_y.T[0]
    #     # plt.scatter(y0_list,y1_list,c='r',s=5, label='x1')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.legend()
    # plt.show()

    ax = plt.axes(projection='3d')

    # x=np.arange(65,110,1)
    # y=np.arange(70,140,1)
    # X, Y = np.meshgrid(x, y)
    # Z=-0.026*(X-Y)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='r')
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        y2_list = []
        for i in range(5,len(y1_list)):
            y2_list.append((y1_list[i]*137 - y1_list[i-1]*300 + y1_list[i-2]*300 - y1_list[i-3]*200 + y1_list[i-4]*75 - y1_list[i-5]*12)/(60*stepsize))
        ax.scatter(y0_list[5:], y1_list[5:], y2_list,c='b',s=5)
    
    

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    plt.legend()
    plt.show()


def case2():
    mode2 = get_mode2(0)
    event1 = get_event1(0)
    y0 = [[99.5, 80], [97.5, 100]]
    y1 = [[100.5, 90], [96, 80]]
    stepsize = 0.1
    maxorder = 1
    boundary_order = 1
    num_mode = 2
    T = 50
    ep = 0.01
    mergeep = 0.01
    method='piecelinear'
    t_list, y_list = simulation_ode_2(mode2, event1, y0, T, stepsize,noise=0)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    np.savetxt("A1.txt",A,fmt='%8f')
    np.savetxt("b1.txt",b,fmt='%8f')
    # P,G,C = infer_model(
    #             t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
    #             num_mode=num_mode, modelist=mode2, event=event1, ep=ep, mergeep = mergeep ,method=method, verbose=False)
    # y1 = [[96, 80], [100.5, 80]]
    t_test_list, y_test_list = simulation_ode_2(mode2, event1, y1, T, stepsize)
    # Y1, Y2 = disc(t_list+t_test_list,y_list+y_test_list)
    # np.savetxt("Y1_1.txt",Y1,fmt='%8f')
    # np.savetxt("Y2_1.txt",Y2,fmt='%8f')
    YT, FT = diff(t_list+t_test_list, y_list+y_test_list, dynamics.mode2t)
    np.savetxt("YT1.txt",YT,fmt='%8f')
    np.savetxt("FT1.txt",FT,fmt='%8f')
    # d_avg = test_model(
                # P, G, C, num_mode, y_list , mode2, event1, maxorder, boundary_order)
    #np.set_printoptions(threshold=np.inf)
    # print(d_avg)
    # print(G)
    # print(C)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # res, drop, clfs = segment_and_fit(A, b1, b2, ytuple)
    # P, G = merge_cluster_tol2(res, A, b1, num_mode, ep)
    # sq_sum = 0
    # posum = 0
    # for j in range(0,1):
    #     A1 = matrowex(A, P[j])
    #     B1 = matrowex(b1, P[j])
    #     clf = linear_model.LinearRegression(fit_intercept=False)
    #     clf.fit(A1, B1)
    #     sq_sum += np.square(clf.predict(A1)-B1).sum()
    #     posum += len(P[j])
    # print(sq_sum)
    # A1 = matrowex(A,P[1])
    # B1 = matrowex(b1,P[1]).getA()[:,0]
    # clf = SVR(kernel='linear',epsilon = 0.00001)
    # clf.fit(A1, B1)
    # print(np.square(clf.predict(A1)-B1).sum())

if __name__ == "__main__":
    # y0 = [[99.5,80],[97.5,100]]
    # t_tuple = [(0,50),(0,50)]
    # stepsize = 0.1
    # maxorder = 1

    # a = case(y0,t_tuple,stepsize,maxorder,mode2,event1,0.01,"new")
    # print(a)
    #case1()
    case2()