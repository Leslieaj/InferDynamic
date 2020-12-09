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
    infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor, \
        diff, infer_model, test_model, segment_and_fit, merge_cluster_tol2, matrowex, seg_droprow

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
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization2, lambda_two_modes3, infer_optimization3, lambda_three_modes, infer_optimizationm
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *

fvdp3_params = [
    [10, Fraction(8,3), 28, 28, 4, 46.92],
]

def get_fvdp3(param_id):
    sigma1,beta1,rho1,sigma2,beta2,rho2= fvdp3_params[param_id]
    
    def fvdp3_1(t,y):
        """Lorenz attractor."""
        y0, y1, y2 = y
        dydt  = [sigma1*(y1-y0), y0*(rho1-y2)-y1, y0*y1-beta1*y2]
        return dydt


    def fvdp3_2(t,y):
        """Lorenz attractor."""
        y0, y1, y2 = y
        dydt  = [sigma2*(y1-y0), y0*(rho2-y2)-y1, y0*y1-beta2*y2]
        return dydt


    return [fvdp3_1,fvdp3_2]


fvdp3 = get_fvdp3(0)

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
    @eventAttr()
    def event1(t,y):
        y0, y1, y2 = y
        return y0 + y1

    return event1

event1 = get_event1(0)

cases = {
    0: {
        'params': 0,
        'y0': [[5,5,5], [2,2,2]],
        'y0_test': [[3,3,3], [4,4,4]],
        't_tuple': 5,
        'stepsize': 0.01,
        'ep': 0.01,
        'mergeep':0.2
    },
    1: {
        'params': 0,
        'y0': [[-3,0,-3]],
        'y0_test': [[5,0,5], [2,0,2]],
        't_tuple': 15,
        'stepsize': 0.002,
        'ep': 0.01,
        'mergeep':0.2
    },
    2: {
        'params': 0,
        'y0': [[-3,0,-3], [2,0,2]],
        'y0_test': [[5,0,5]],
        't_tuple': 10,
        'stepsize': 0.002,
        'ep': 0.01,
        'mergeep':0.4
    },
    3: {
        'params': 0,
        'y0': [[5,5,5], [2,2,2], [3,3,3], [2,2,-2], [1,0,1]],
        'y0_test': [[4,4,4], [3,-2,2]],
        't_tuple': 5,
        'stepsize': 0.004,
        'ep': 0.01,
        'mergeep':0.4
    }
}



def case(y0,t_tuple,stepsize,maxorder,modelist,event,ep,method):
    t_list, y_list = simulation_ode_2(modelist, event, y0, t_tuple, stepsize)
    
    
    if method == "new":
        
        
        A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
        P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
        P,D=dropclass(P,G,D,A,b,Y,0.01,stepsize)
        # print(len(P))
        print(G)
        y = []
        x = []

        for j in range(0,len(P[0])):
            y.append(1)
            x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1], 3:Y[P[0][j],2]})
        
        for j in range(0,len(P[1])):
            y.append(-1)
            x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1], 3:Y[P[1][j],2]})

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
        for i in range(0,nsv):
            a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
            a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
            a3 = a3 + svc[i][0] * 0.5 * sv[i][3]
            g = g + svc[i][0]
        
        print(a1/a1,a2/a1,a3/a1,g/a1)
        def f(x):
            return a1*x[0] + a2*x[1] + a3*x[2] + g > 0

        @eventAttr()
        def eventtest(t,y):
            y0, y1, y2 = y
            return a1 * y0 + a2 * y1 + a3 * y2 + g
        ttest_list, ytest_list = simulation_ode_2([ode_test(G[0],maxorder),ode_test(G[1],maxorder)], eventtest, y0, t_tuple, stepsize)
        ax = plt.axes(projection='3d')
        for temp_y in y_list[0:1]:
            y0_list = temp_y.T[0]
            y1_list = temp_y.T[1]
            y2_list = temp_y.T[2]
            ax.plot3D(y0_list, y1_list, y2_list,c='b')
        for temp_y in ytest_list[0:1]:
            y0_list = temp_y.T[0]
            y1_list = temp_y.T[1]
            y2_list = temp_y.T[2]
            ax.plot3D(y0_list, y1_list, y2_list,c='r')
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

def case2():
    y0 = [[5,5,5], [2,2,2]]
    stepsize = 0.01
    maxorder = 2
    boundary_order = 1
    num_mode = 2
    T = 5
    ep = 0.01
    mergeep = 0.01
    method='merge'
    

    t_list, y_list = simulation_ode_2(get_fvdp3(0), get_event1(0), y0, T, stepsize)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    np.savetxt("data/A2.txt",A,fmt='%8f')
    np.savetxt("data/b2.txt",b,fmt='%8f')
    # print(y_list)

    # P,G,C = infer_model(
    #             t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
    #             num_mode=num_mode, modelist=fvdp3, event=event1, ep=ep, mergeep= mergeep, method=method, verbose=False)
    
    y1 = [[3,3,3], [4,4,4]]
    t_test_list, y_test_list = simulation_ode_2(get_fvdp3(0), get_event1(0), y0, T, stepsize)
    YT, FT = diff(t_list+t_test_list, y_list+y_test_list, dynamics.fvdp3_3)
    # np.savetxt("data/YT"+str(n)+".txt",YT,fmt='%8f')
    # np.savetxt("data/FT"+str(n)+".txt",FT,fmt='%8f')
    np.savetxt("data/YT2.txt",YT,fmt='%8f')
    np.savetxt("data/FT2.txt",FT,fmt='%8f')
    # d_avg = test_model(
    #             P, G, C, num_mode, y_list + y_test_list, fvdp3, event1, maxorder, boundary_order)
    # print(d_avg)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # res, drop, clfs = segment_and_fit(A, b1, b2, ytuple)
    # P, G = merge_cluster_tol2(res, A, b1, num_mode, ep)
    # sq_sum = 0
    # posum = 0
    # for j in range(0,num_mode):
    #     A1 = matrowex(A, P[j])
    #     B1 = matrowex(b1, P[j])
    #     clf = linear_model.LinearRegression(fit_intercept=False)
    #     clf.fit(A1, B1)
    #     sq_sum += np.square(clf.predict(A1)-B1).sum()
    #     posum += len(P[j])
    # print(sq_sum/posum)


def case3():
    y0 = [[5,5,5], [2,2,2]]
    stepsize = 0.01
    maxorder = 2
    boundary_order = 1
    num_mode = 2
    T = 5
    ep = 0.01
    mergeep = 0.01
    method='merge'
    

    t_list, y_list = simulation_ode_2(get_fvdp3(0), get_event1(0), y0, T, stepsize)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    x0 = np.zeros(num_mode*A.shape[1]*b.shape[1])
    re = infer_optimizationm(x0, A, b, num_mode)
    print(re.fun)
    print(re.success)
    print(re.x)
    A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    A, b1, b2 = seg_droprow(A,b1,b2,ep)
    x0 = np.zeros(num_mode*A.shape[1]*b1.shape[1])
    re = infer_optimizationm(x0, A, b1, num_mode)
    print(re.fun)
    print(re.success)
    print(re.x)
    # P,G,C = infer_model(
    #             t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
    #             num_mode=num_mode, modelist=fvdp3, event=event1, ep=ep, mergeep= mergeep, method=method, verbose=False)
    
    # y1 = [[3,3,3], [4,4,4]]
    # t_test_list, y_test_list = simulation_ode_2(get_fvdp3(0), get_event1(0), y0, T, stepsize)
    # YT, FT = diff(t_list+t_test_list, y_list+y_test_list, dynamics.fvdp3_3)

if __name__ == "__main__":
    # case2()
    case3()