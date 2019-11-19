import numpy as np
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from dynamics import fvdp2_1, fvdp3_1
from infer_multi import simulation_ode, infer_dynamic
from draw import draw, draw2D, draw3D

def case1():
    y0 = [[a,b] for a in np.arange(-0.5,0.5+0.25,0.25) for b in np.arange(-2.5,-1.5+0.25,0.25)]
    t_tuple = (0,1)
    stepsize = 0.001
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(fvdp2_1, y0, t_tuple, stepsize)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order, eps=0)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)
    draw2D(y_list)

def case2():
    y0 = [[-8,7,27]]
    t_tuple = (0,100)
    stepsize = 0.001
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(fvdp3_1, y0, t_tuple, stepsize)
    end_simulation = time.time()
    result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_points, y_list, stepsize, order, eps=0)
    end_inference = time.time()

    print(result_coef)
    print()
    print("Total time: ", end_inference-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", calcdiff_time)
    print("Pseudoinv time: ", pseudoinv_time)

    draw3D(y_list)


if __name__ == "__main__":
    # case1()
    case2()
