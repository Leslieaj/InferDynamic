import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
from dynamics import dydx3, fvdp2_1, fvdp3_1, mode2_1, mode2_1_test, conti_test, conti_test_test, conti_test1, ode_test
from infer_multi_ch import simulation_ode, infer_dynamic, parti, infer_dynamic_modes, infer_dynamic_modes_ex, infer_dynamic_modes_exx, dist, diff_method

import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *


def case1():
    y0 = [[0,7]]
    t_tuple = [(0,30)]
    stepsize = 0.01
    order = 2

    start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.1)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][-1])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()

    
def case2():
    y0 = [[1,3],[-1,-2],[-3,-5],[2,4],[-2,3],[-4,5],[4,7],[-2,-10],[5,8]]
    t_tuple = [(0,2),(0,3),(0,4),(0,5),(0,1),(0,1),(0,5),(0,6),(0,6)]
    stepsize = 0.01
    order = 2
    maxorder = 6
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    # modes, coefs = infer_dynamic_modes(t_list, y_list, stepsize, maxorder, 0.01)
    modes, coefs, mdors = infer_dynamic_modes_ex(t_list, y_list, stepsize, maxorder, 0.01)
    print(modes)
    print(coefs)
    print(mdors)
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()



def case3():
    y0 = [[9]]
    t_tuple = [(0,3)]
    stepsize = 0.01
    order = 2
    t_list, y_list = simulation_ode(dydx3, y0, t_tuple, stepsize, eps=0)
    for i in range(0,len(y_list)):
        plt.plot(t_list[i],y_list[i])
    plt.show()


def case4():
    y0 = [[0,0],[1,0]]
    t_tuple = [(0,2.5),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 4

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

    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    print(modes)
    print(coefs)
    print(mdors)

def case5():
    y0 = [[0,0],[1,0]]
    t_tuple = [(0,2.5),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 4

    
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)

    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # print(modes)
    # print(coefs)
    # print(mdors)
    ttest_list=[]
    ttest_list.append(tpar_list[0])
    ttest_list.append(tpar_list[1])
    ytest_list=[]
    ytest_list.append(ypar_list[0])
    ytest_list.append(ypar_list[1])
    A, b = diff_method(ttest_list, ytest_list, 4, stepsize)
    g = pinv2(A).dot(b)
    print(g.T)
    t_start = tpar_list[0][0]
    t_end = tpar_list[0][-1]
    t_start = 0
    t_end = 0.5
    t_points = np.arange(t_start, t_end + stepsize, stepsize)
    y_object = solve_ivp(ode_test(g.T,4), (t_start, t_end+stepsize), ypar_list[0][0], t_eval = t_points, rtol=1e-7, atol=1e-9)
    y_points = y_object.y.T
    print(y_points)


if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    # case4()
    case5()
    
