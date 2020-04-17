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
    infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor

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



modetr_params = [
    [0.5,0.5,-9,3,5,-0.1,-10,-7,-1],
]

def get_modetr(param_id):
    a1,a2,a3,a4,b1,b2,b3,c1,c2 = modetr_params[param_id]

    def modetr_1(t,y):
        y0, y1= y
        dydt = dydt = [a1*y0**2 + a2*y1, a3*y0+a4]
        return dydt

    def modetr_2(t,y):
        y0, y1= y
        dydt = dydt = [b1, b2*y0+b3]
        return dydt

    def modetr_3(t,y):
        y0, y1= y
        dydt = dydt = [c1 , c2*y0]
        return dydt

    return [modetr_1,modetr_2,modetr_3]

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
        y0, y1 = y
        return y0

    @eventAttr()
    def eventtr_2(t,y):
        y0, y1 = y
        return y1

    return [eventtr_1, eventtr_2, eventtr_2]


eventtr_1, eventtr_2, _ = get_event(0)

def get_labeltest(param_id):
    def labeltest(y):
        if eventtr_1(0,y)<0 and eventtr_2(0,y)>0:
            return 0
        elif eventtr_1(0,y)>=0 and eventtr_2(0,y)>0:
            return 1
        else:
            return 2
    return labeltest

labeltest = get_labeltest(0)

cases = {
    0: {
        'params': 0,
        'y0': [[-1,1],[1,4],[2,-3]],
        't_tuple': [(0,5),(0,5),(0,5)],
        'stepsize': 0.01,
        'ep': 0.01
    },
    1: {
        'params': 0,
        'y0': [[-1,1],[1,4]],
        't_tuple': [(0,5),(0,5)],
        'stepsize': 0.01,
        'ep': 0.01
    },
    2: {
        'params': 0,
        'y0': [[-1,1],[1,4],[2,-3],[1,1],[3,1]],
        't_tuple': [(0,5),(0,5),(0,5),(0,5),(0,5)],
        'stepsize': 0.01,
        'ep': 0.01,
    },
    3: {
        'params': 0,
        'y0': [[-1,1],[1,4],[2,-3],[1,1],[3,1]],
        't_tuple': [(0,5),(0,5),(0,5),(0,5),(0,5)],
        'stepsize': 0.01,
        'ep': 0.002,
    },
}

def case(y0,t_tuple,stepsize,maxorder,modelist,eventlist,labeltest,ep,method):
    t_list, y_list = simulation_ode_3(modelist, eventlist, labeltest, y0, t_tuple, stepsize)

    if method == "new":
        A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
        P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep)
        P,G = reclass(A,b,P,ep)
        P,D = dropclass(P,G,D,A,b,Y,ep,stepsize)
        print(G)

        for i in range(0,len(P)):
            y0_list = []
            y1_list = []
            for j in range(0,len(P[i])):
                y0_list.append(Y[P[i][j],0])
                y1_list.append(Y[P[i][j],1])
        
            plt.scatter(y0_list,y1_list,s=1)
        plt.show()

        y=[]
        x=[]

        for j in range(0,len(P[2])):
            y.append(1)
            x.append({1:Y[P[2][j],0], 2:Y[P[2][j],1]})
        
        for j in range(0,len(P[1])):
            y.append(-1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})
        
        for j in range(0,len(P[0])):
            y.append(-1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})

        prob  = svm_problem(y, x)
        param = svm_parameter('-t 1 -d 1 -c 100 -r 1 -b 0 -q')
        m = svm_train(prob, param)
        svm_save_model('model_file1', m)
        nsv = m.get_nr_sv()
        svc = m.get_sv_coef()
        sv = m.get_SV()
        g = -m.rho[0]
        a1 = 0
        a2 = 0
        for i in range(0,nsv):
            a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
            a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
            g = g + svc[i][0]*1
        print(a1/a2,a2/a2,g/a2)
        def f(x):
            g = -m.rho[0]
            for i in range(0,nsv):
                g = g + svc[i][0] * (0.5*(x[0]*sv[i][1]+x[1]*sv[i][2])+1)
            return g>0

        x=[]
        y=[]

        for j in range(0,len(P[1])):
            y.append(1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})
        
        for j in range(0,len(P[0])):
            y.append(-1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})

        prob  = svm_problem(y, x)
        param = svm_parameter('-t 1 -d 1 -c 100 -r 1 -b 0 -q')
        n = svm_train(prob, param)
        svm_save_model('model_file2', n)
        # p_label, p_acc, p_val = svm_predict(y, x, n)
        nsv1 = n.get_nr_sv()
        svc1 = n.get_sv_coef()
        sv1 = n.get_SV()
        g1 = -n.rho[0]
        b1 = 0
        b2 = 0
        for i in range(0,nsv1):
            b1 = b1 + svc1[i][0] * 0.5 * sv1[i][1]
            b2 = b2 + svc1[i][0] * 0.5 * sv1[i][2]
            g1 = g1 + svc1[i][0]*1
        print(b1/b1,b2/b1,g1/b1)
        def h(x):
            g = -n.rho[0]
            for i in range(0,nsv1):
                g = g + svc1[i][0] * (0.5*(x[0]*sv1[i][1]+x[1]*sv1[i][2])+1)
            return g > 0

        @eventAttr()
        def eventtest1(t,y):
            y0, y1 = y
            return a1 * y0 + a2 * y1 + g
        @eventAttr()
        def eventtest2(t,y):
            y0, y1 = y
            return b1 * y0 + b2 * y1 + g1

        def labeltesttest(y):
            if eventtest1(0,y)>0:
                return 2
            elif eventtest2(0,y)>0:
                return 1
            else:
                return 0
        
        ttest_list, ytest_list = simulation_ode_3([ode_test(G[0],maxorder),ode_test(G[1],maxorder),ode_test(G[2],maxorder)], [eventtest1,eventtest2], labeltesttest, y0, t_tuple, stepsize)

        for temp_y in y_list:
            y0_list = temp_y.T[0]
            y1_list = temp_y.T[1]
            plt.plot(y0_list,y1_list,c='b')
        for temp_y in ytest_list:
            y0_list = temp_y.T[0]
            y1_list = temp_y.T[1]
            plt.plot(y0_list,y1_list,c='r')
        plt.show()
    
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
            exact = modelist[labeltest(ypoints[i])](0,ypoints[i])
            if f(ypoints[i]) == 1:
                predict = np.matmul(get_poly_pt(ypoints[i]),G[2].T)
            elif h(ypoints[i]) == 1:
                predict = np.matmul(get_poly_pt(ypoints[i]),G[1].T)
            else:
                predict = np.matmul(get_poly_pt(ypoints[i]),G[0].T)

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


if __name__ == "__main__":
    y0 = [[-1,1],[1,4],[2,-3]]
    t_tuple = [(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    eventlist=[eventtr_1,eventtr_2]
    a = case(y0,t_tuple,stepsize,maxorder,modetr,eventlist,labeltest,0.01,"new")
    print(a)
