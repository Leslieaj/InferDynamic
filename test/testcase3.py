import numpy as np
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from dynamics import mode2_1
from infer_single import simulation_ode
from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3

def case1():
    y0 = [[0,3]]
    t_tuple = (0,20)
    stepsize = 0.01
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()

    final_A_mat, final_b_mat = get_coef(t_points, y_list, order, stepsize)
    end_coedf = time.time()

    # x0 = np.ones((2*final_A_mat.shape[1], final_b_mat.shape[1]))*0.1
    # x0 = np.ones((1,2*final_A_mat.shape[1]*final_b_mat.shape[1]))*0.1
    print(2*final_A_mat.shape[1]*final_b_mat.shape[1])
    x0 = np.random.uniform(-1,1,[2*final_A_mat.shape[1]*final_b_mat.shape[1]])
    # print(x0)
    x1 = np.array([0,0,0,-0.26,0.26,0,0,0,0, 0,0,1.0, 0,0,0,-0.26,0.26,0, 0,0,0, 0,0,-1.0])

    pr = cProfile.Profile()
    pr.enable()
    results = infer_optimization(x0, final_A_mat, final_b_mat)
    p = Stats(pr)
    p.strip_dirs()
    p.sort_stats('cumtime')
    p.print_stats(100)
    
    end_optimization = time.time()
    # print(results)
    print(results.x)
    print(lambda_two_modes(final_A_mat,final_b_mat)(x0))
    print(lambda_two_modes(final_A_mat,final_b_mat)(results.x))
    print(lambda_two_modes(final_A_mat,final_b_mat)(x1))
    print(results.success)
    print(results.message)


    print("Simulation time: ", end_simulation-start)
    print("Optimazation time: ", end_optimization-end_coedf)
    # draw2D_dots(y_list)

    A_row = final_A_mat.shape[0]
    A_col = final_A_mat.shape[1]
    b_col = final_b_mat.shape[1]
    x1, x2 = np.array_split(results.x,2)
    x1 = np.mat(x1.reshape([A_col,b_col],order='F'))
    y1 = np.matmul(final_A_mat,x1) - final_b_mat
    y1 = np.multiply(y1,y1)
    y1 = y1.sum(axis=1)
    x2 = np.mat(x2.reshape([A_col,b_col],order='F'))
    y2 = np.matmul(final_A_mat,x2) - final_b_mat
    y2 = np.multiply(y2,y2)
    y2 = y2.sum(axis=1)
    modet = np.zeros(A_row)
    for i in range(0,A_row):
        if y1[i] < y2[i]:
            modet[i] = 1
        else:
            modet[i] = 2
    plt.plot(t_points,y_list[0])   
    plt.plot(t_points[:A_row],modet)
    plt.show() 


if __name__ == "__main__":
    case1()