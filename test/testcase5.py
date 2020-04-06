import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
from dynamics import dydx3, fvdp2_1, fvdp3_1, mode2_1, mode2_11, mode2_1_test, conti_test, conti_test_test, conti_test1, ode_test
from infer_multi_ch import simulation_ode, simulation_ode_stiff, infer_dynamic, parti, infer_dynamic_modes_ex, infer_dynamic_modes_exx, dist, diff_method, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie, infer_dynamic_modes_new, diff_method_new1, diff_method_new

import dynamics
import warnings
warnings.filterwarnings('ignore')

import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *




# def final_infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1):
#     A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
#     P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep)
#     for i in range(0,len(P)):
        



def case1():
    # y0 = [[1,3],[-1,-2],[-3,-5],[2,4]]
    # t_tuple = [(0,20),(0,10),(0,15),(0,15)]
    y0 = [[1,3],[-1,-2]]
    t_tuple = [(0,20),(0,10)]
    stepsize = 0.01
    order = 2
    maxorder = 2
    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    
    A, b, Y = diff_method_new1(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    
    print(P)
    print(G)
    print(D)
    y = []
    x = []

    for j in range(0,len(P[0])):
        y.append(1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})
    
    for j in range(0,len(P[1])):
        y.append(-1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 3 -c 10 -b 0 -r 1.5')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    yt = [-1,-1,1,1]
    xt = [{1:1, 2:1},{1:-1, 2:1},{1:1, 2:-1},{1:-1, 2:-1}]
    print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # print(p_label)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    def clafun(x,y):
        g = -m.rho[0]
        for i in range(0,nsv):
            g = g + svc[i][0] * ((0.5 * (x*sv[i][1] + y*sv[i][2]) + 1.5)**3)
        return np.sign(g)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-3,3,0.1)
    y = np.arange(-3,3,0.1)
    X,Y = np.meshgrid(x,y)#创建网格，这个是关键
    Z = clafun(X,Y)
    plt.xlabel('x')
    plt.ylabel('y')
 
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    plt.show()
    


def case2():
    y0 = [[0,0],[1,0],[0.3,0],[2.7,0]]
    t_tuple = [(0,2.5),(0,2),(0,2),(0,1)]
    stepsize = 0.001
    order = 2
    maxorder = 2

    # start = time.time()
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    print(P)
    print(G)
    print(D)


if __name__ == "__main__":
    case1()
    # case2()