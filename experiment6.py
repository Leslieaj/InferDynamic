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
    infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor, infer_model,\
        diff, test_model, merge_cluster_tol2, diff_method_backandfor, matrowex, segment_and_fit, dropclass0

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
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization2, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *



modetr_params = [
    [0.5,0.7,1.0,1.46,1.85,0.1,-0.2,0.3,-0.4,0.5],
]

def get_modetr(param_id):
    a,b,c,d,e,m1,m2,m3,m4,m5 = modetr_params[param_id]

    def modetr_1(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        dydt = dydt = [a*y01, 0-a*y00, b*y11, 0-b*y10, c*y21, 0-c*y20, d*y31, 0-d*y30, e*y41, 0-e*y40, 1 + m1 * y00**4]
        return dydt

    def modetr_2(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        dydt = dydt = [a*y01, 0-a*y00, b*y11, 0-b*y10, c*y21, 0-c*y20, d*y31, 0-d*y30, e*y41, 0-e*y40, 1 + m2 * y10**4]
        return dydt

    def modetr_3(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        dydt = dydt = [a*y01, 0-a*y00, b*y11, 0-b*y10, c*y21, 0-c*y20, d*y31, 0-d*y30, e*y41, 0-e*y40, 1 + m3 * y20**4]
        return dydt

    def modetr_4(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        dydt = dydt = [a*y01, 0-a*y00, b*y11, 0-b*y10, c*y21, 0-c*y20, d*y31, 0-d*y30, e*y41, 0-e*y40, 1 + m4 * y30**4]
        return dydt
    
    def modetr_5(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        dydt = dydt = [a*y01, 0-a*y00, b*y11, 0-b*y10, c*y21, 0-c*y20, d*y31, 0-d*y30, e*y41, 0-e*y40, 1 + m5 * y40**4]
        return dydt

    return [modetr_1,modetr_2,modetr_3,modetr_4,modetr_5]

modetr = get_modetr(0)

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
    def eventtr_1(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        return z-1

    @eventAttr()
    def eventtr_2(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        return z-2

    @eventAttr()
    def eventtr_3(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        return z-3

    @eventAttr()
    def eventtr_4(t,y):
        y00, y01, y10, y11, y20, y21, y30, y31, y40, y41, z = y
        return z-4

    
    return [eventtr_1, eventtr_2, eventtr_3, eventtr_4]


eventtr_1, eventtr_2, eventtr_3, eventtr_4= get_event(0)

def get_labeltest(param_id):
    def labeltest(y):
        if eventtr_1(0,y)<0:
            return 0
        elif eventtr_1(0,y)>=0 and eventtr_2(0,y)<0:
            return 1
        elif eventtr_2(0,y)>=0 and eventtr_3(0,y)<0:
            return 2
        elif eventtr_3(0,y)>=0 and eventtr_4(0,y)<0:
            return 3
        else:
            return 4
    return labeltest

labeltest = get_labeltest(0)

def case1():
    np.random.seed(0)
    modetr = get_modetr(0)
    event = get_event(0)
    labeltest = get_labeltest(0)
    y0 = [[0,1.2,0,1.2,0,1.2,0,1.2,0,1.2,1.1],[1,0,1,0,1,0,1,0,1,0,0],[-1,0,-1,0,-1,0,-1,0,-1,0,2.1],[0,-1.2,0,-1.2,0,-1.2,0,-1.2,0,-1.2,3.1]]
    # y1 = [[3,-1], [-1,3]]
    T = 2
    stepsize = 0.002
    maxorder = 4
    boundary_order = 1
    num_mode = 5
    ep = 0.02
    mergeep = 0.01
    t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # res, drop, clfs = segment_and_fit(A, b1, b2, ytuple,ep=0.005)
    # np.savetxt("data/YY.txt",Y,fmt='%8f')
    # print(len(res))
    # print(res)
    # P, G = merge_cluster_tol2(res, A, b1, num_mode, ep)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    np.savetxt("data/YY.txt",Y,fmt='%8f')
    P, G, D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep)
    print(P)
    print(len(P))
    if len(P)>num_mode:
        P, G = merge_cluster_tol2(P, A, b, num_mode,0.01)
    P, _ = dropclass0(P, G, D, A, b, Y, ep, stepsize)
    print(P)
    print(len(P))

    sq_sum = 0
    posum = 0
    for j in range(0,num_mode):
        A1 = matrowex(A, P[j])
        B1 = matrowex(b, P[j])
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(A1, B1)
        sq_sum += np.square(clf.predict(A1)-B1).sum()
        posum += len(P[j])
    print(sq_sum/posum)

if __name__ == "__main__":
    case1()