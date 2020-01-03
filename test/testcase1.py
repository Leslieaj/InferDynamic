# import numpy as np
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from dynamics import dydx1, dydx2, fvdp2, ode_test
from infer_single import simulation_ode, infer_dynamic
from draw import draw, draw2D
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def case1():
    y0 = [[5],[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,4)
    stepsize = 0.01
    order = 3
    start = time.time()
    t_points, y_list = simulation_ode(dydx1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    draw(t_points, y_list)

def case2():
    y0 = [[20]]
    t_tuple = (0,4)
    stepsize = 0.001
    order = 8
    start = time.time()
    t_points, y_list = simulation_ode(dydx2, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()
    t_points, y_list_test = simulation_ode(ode_test(result_coef,order), y0, t_tuple, stepsize, eps=0)
    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    for temp_y in y_list:
        plt.scatter(t_points,temp_y,s=0.1,c='r')
    for temp_y in y_list_test:
        plt.scatter(t_points,temp_y,s=0.1,c='b')
    plt.show()

def case3():
    y0 = [[5,-3],[2,0],[-2,3]]
    t_tuple = (0,10)
    stepsize = 0.01
    order = 3
    start = time.time()
    t_points, y_list = simulation_ode(fvdp2, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    draw2D(y_list)
    # draw(t_points, y_list)

if __name__ == "__main__":
    # case1()
    case2()
    # case3()
