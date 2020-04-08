import numpy as np
from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
# import random

from dynamics import dydx1, dydx2, fvdp2, fvdp2_1
from generator import generate_complete_polynomial
from draw import draw, draw2D

def simulation_ode(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0):
    """ Given a ODE function, some initial state, stepsize, then return the points.
        @ode_func: ODE function 
        @y0: inital state
        @t_tuple: 
        @stepsize: step size
        @eps: guass noise (defult 0)
    """
    t_start = t_tuple[0]
    t_end = t_tuple[1]
    t_points = np.arange(t_start, t_end + stepsize, stepsize)
    y_list = []

    for k in range(0,len(y0)):
        y_object = solve_ivp(ode_func, (t_start, t_end+stepsize), y0[k], t_eval = t_points, rtol=1e-7, atol=1e-9)
        y_points = y_object.y.T
        if eps > 0:
            for i in range(0,y_points.shape[0]):
                for j in range(0,y_points.shape[1]):
                    y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
        y_list.append(y_points)
    return t_points, y_list

def diff_method(t_points, y_list, order, stepsize):
    """Using multi-step difference method (Adams5) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    L_t = len(t_points)
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomial(L_y,order)
    L_p = gene.shape[0]

    D = L_t - 4    #Adams5

    for k in range(0,len(y_list)):
        y_points = y_list[k]
        A_matrix = np.zeros((D*2, L_p), dtype=np.double)
        b_matrix = np.zeros((D*2, L_y),  dtype=np.double)
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        # Adams5
        for i in range(0, 2*D):
            if i < D:     # forward
                A_matrix[i] = (-19*coef_matrix[i]+106*coef_matrix[i+1]-264*coef_matrix[i+2]+646*coef_matrix[i+3]+251*coef_matrix[i+4])*stepsize/720
                b_matrix[i] = y_points[i+4]-y_points[i+3]
            else:         # backward
                A_matrix[i] = (-19*coef_matrix[i-D+4]+106*coef_matrix[i-D+3]-264*coef_matrix[i-D+2]+646*coef_matrix[i-D+1]+251*coef_matrix[i-D])*stepsize/720
                b_matrix[i] = y_points[i-D+1]-y_points[i-D]
        
        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]

    return final_A_mat, final_b_mat

def infer_dynamic(t_points, y_list, stepsize, order):
    """ The main function to infer a dynamic system.
    """
    start = time.time()
    A, b = diff_method(t_points, y_list, order, stepsize)
    end_diff = time.time()
    # Moore-Penrose Inverse (pseudoinverse)
    # g = linalg.pinv(A).dot(b) # using least-square solver
    g = pinv2(A).dot(b) # using using its singular-value decomposition and including all 'large' singular values.
    end_pseudoinv = time.time()
    return g.T, end_diff-start, end_pseudoinv-end_diff


if __name__ == "__main__":
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
