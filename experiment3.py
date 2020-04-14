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

mode_params = [
    [0.1,0.04,-0.9,-0.1,0.06,-0.7],
]

def get_mode(param_id):
    a1, b1,c1, a2, b2,c2 = mode_params[param_id]

    def mode1(t,y):
        y0, y1 = y
        dydt = [a1*y0**2 + b1*y1**3, c1*y0]
        return dydt
    

    def mode2(t,y):
        y0, y1 = y
        dydt = [a2*y0**2 + b2*y1**3, c2*y0]
        return dydt
    return [mode1,mode2]

mode = get_mode(0)


def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator


@eventAttr()
def event1(t,y):
    y0, y1 = y
    return y1 - 0.2*y0**2



def case(y0,t_tuple,stepsize,maxorder,modelist,event,ep,method):
    t_list, y_list = simulation_ode_2(modelist, event, y0, t_tuple, stepsize)
    
    
    if method == "new":
        
        
        A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
        P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
        P,D=dropclass(P,G,D,A,b,Y,0.01,stepsize)
        # print(len(P))
        # print(G)
        

        y = []
        x = []

        for j in range(0,len(P[0])):
            y.append(1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})
        
        for j in range(0,len(P[1])):
            y.append(-1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})

        prob  = svm_problem(y, x)
        param = svm_parameter('-t 1 -d 2 -r 1 -c 10 -b 0 -q')
        m = svm_train(prob, param)
        svm_save_model('model_file', m)
        # p_label, p_acc, p_val = svm_predict(y, x, m)
        nsv = m.get_nr_sv()
        svc = m.get_sv_coef()
        sv = m.get_SV()
        def f(x):
            g = -m.rho[0]
            for i in range(0,nsv):
                g = g + svc[i][0] * (0.5*(x[0]*sv[i][1]+x[1]*sv[i][2])+1)**2
            return g>0

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


if __name__ == "__main__":
    y0 = [[0,1],[0,2],[0,3],[0,4],[0,5]]
    t_tuple = [(0,20),(0,20),(0,20),(0,20),(0,20)]
    stepsize = 0.01
    maxorder = 3
    a = case(y0,t_tuple,stepsize,maxorder,mode,event1,0.01,"new")
    print(a)
