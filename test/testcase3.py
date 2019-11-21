import numpy as np
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from dynamics import mode2_1
from infer_single import simulation_ode
from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization

def case1():
    y0 = [[0,3]]
    t_tuple = (0,10)
    stepsize = 0.01
    order = 1

    start = time.time()
    t_points, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()

    final_A_mat, final_b_mat = get_coef(t_points, y_list, order, stepsize)
    end_coedf = time.time()

    x0 = np.ones((2*final_A_mat.shape[1], final_b_mat.shape[1]))*0.1
    results = infer_optimization(x0, final_A_mat, final_b_mat)
    end_optimization = time.time()

    print(results.x)
    print(results.success)
    print(results.message)


    print("Simulation time: ", end_simulation-start)
    print("Optimazation time: ", end_optimization-end_coedf)
    draw2D_dots(y_list)

if __name__ == "__main__":
    case1()