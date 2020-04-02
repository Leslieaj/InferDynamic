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
from dynamics import dydx3, fvdp2_1, fvdp3_1, mode2_1, mode2_11, mode2_1_test, conti_test, conti_test_test, conti_test1, ode_test
from infer_multi_ch import simulation_ode, simulation_ode_stiff, infer_dynamic, parti, infer_dynamic_modes_ex, infer_dynamic_modes_exx, dist, diff_method, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie, infer_dynamic_modes_new

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



def case1():
    y0 = [[1,7]]
    t_tuple = [(0,25)]
    stepsize = 0.01
    order = 2
    maxorder = 1
    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.005)
    print(P)
    print(G)
    print(D)


if __name__ == "__main__":
    case1()