import numpy as np
import matplotlib.pyplot as plt
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from dynamics import fvdp2_1, fvdp3_1, mode2_1, mode2_1_test, conti_test, conti_test_test, conti_test1
from infer_multi import simulation_ode, infer_dynamic
from draw import draw, draw2D, draw3D

def case1():
    y0 = [[a,b] for a in np.arange(-0.5,0.5+0.25,0.25) for b in np.arange(-2.5,-1.5+0.25,0.25)]
    t_tuple = (0,1)
    stepsize = 0.001
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(fvdp2_1, y0, t_tuple, stepsize, eps=0)
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

def case2():
    y0 = [[a,b] for a in np.arange(-0.5,0.5+0.25,0.25) for b in np.arange(-2.5,-1.5+0.25,0.25)]
    t_tuple = (0,1)
    stepsize = 0.001
    order = 2

    start = time.time()
    y_list_20 = []
    y_list_ave = []
    t_points = []
    for i in range(0,20):
        t_points, y_list = simulation_ode(fvdp2_1, y0, t_tuple, stepsize, eps=0.01)
        y_list_20.append(y_list)
    
    for j in range(0, len(y_list_20[0])):
        y_ppoints = np.zeros((y_list_20[0][0].shape[0],y_list_20[0][0].shape[1]))
        for i in range(0,20):
            y_ppoints = y_ppoints + y_list_20[i][j]
        y_ppoints = y_ppoints/20.0
        y_list_ave.append(y_ppoints)

    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list_ave, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    draw2D(y_list_ave)

def case3():
    y0 = [[-8,7,27]]
    t_tuple = (0,100)
    stepsize = 0.001
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(fvdp3_1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)

    draw3D(y_list)

def case4():
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
    # draw2D(y_list)

    # ode1_coef = result_coef[0]
    # ode2_coef = result_coef[1]
    
    # return result_coef, y_list 


def case5():
    y0 = [[1,0],[0.4,0],[0.5,0],[1.4,0],[2.6,0],[6.0,0]]
    t_tuple = (0,3)
    stepsize = 0.01
    order = 6

    start = time.time()
    t_points, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    t_points, y_list_test = simulation_ode(conti_test_test(result_coef,order), y0, t_tuple, stepsize, eps=0)
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        # plt.scatter(y0_list,y1_list,s=1,c='b')
        plt.plot(y0_list,y1_list,c='b')
    for temp_y in y_list_test:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        # plt.scatter(y0_list,y1_list,s=1,c='r')
        plt.plot(y0_list,y1_list,c='r')
    plt.show()


def case6():
    y0 = [[0.1,0],[0.4,0],[1.3,0],[1.7,0]]
    t_tuple = (0,4)
    stepsize = 0.001
    order = 4

    start = time.time()
    t_points, y_list = simulation_ode(conti_test1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    t_points, y_list_test = simulation_ode(conti_test_test(result_coef,order), y0, t_tuple, stepsize, eps=0)
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.scatter(y0_list,y1_list,s=0.1,c='b')
        # plt.plot(y0_list,y1_list,c='b')
    for temp_y in y_list_test:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.scatter(y0_list,y1_list,s=0.1,c='r')
        # plt.plot(y0_list,y1_list,c='r')
    plt.show()

if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    # case4()
    case5()
    # case6()
