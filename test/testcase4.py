import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from dynamics import fvdp2_1, fvdp3_1, mode2_1, mode2_1_test, conti_test, conti_test_test, conti_test1
from infer_multi import simulation_ode, infer_dynamic

import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *


def case1():
    y0 = [[0,3],[0,5]]
    t_tuple = (0,20)
    stepsize = 0.01
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)

    t_points, y_list_test = simulation_ode(mode2_1_test(result_coef,order), y0, t_tuple, stepsize, eps=0)
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    for temp_y in y_list_test:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'r')
    plt.show()





if __name__ == "__main__":
    case1()
    
