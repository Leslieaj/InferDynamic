import numpy as np
from scipy import linalg, linspace
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
import random

from dynamics import dydx1, dydx2, fvdp2
from generator import generate_complete_polynomail

def simulation_ode(ode_func, y0, t_tuple, stepsize):
    """ Given a ODE function, some initial state, stepsize, then return the points.
        @ode_func: ODE function 
        @y0: inital state
        @t_tuple: 
        @stepsize: step size
    """
    t_start = t_tuple[0]
    t_end = t_tuple[1]
    t_points = np.arange(t_start, t_end + stepsize, stepsize)
    y_list = []
    eps = 0.01
    for k in range(0,len(y0)):
        y_object = solve_ivp(ode_func, (t_start, t_end+stepsize), y0[k], t_eval = t_points, rtol=1e-7, atol=1e-9)
        y_points = y_object.y.T
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
    gene = generate_complete_polynomail(L_y,order)
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

def draw(t,y):
    """Draw
    """
    for temp_y in y:
        plt.plot(t,temp_y)
    plt.show()
    return 0

def infer_dynamic():
    """ The main function to infer a dynamic system.
    """
    y0 = [[5,-3],[2,0],[-2,3]]
    # y0 = [[5],[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,4)
    stepsize = 0.01

    start = time.time()
    t_points, y_list = simulation_ode(fvdp2, y0, t_tuple, stepsize)
    # t_points, y_list = simulation_ode(dydx2, y0, t_tuple, stepsize)
    end_simulation = time.time()
    A, b = diff_method(t_points, y_list, 3, stepsize)
    end_diff = time.time()
    g = linalg.pinv(A).dot(b)
    end_pseudoinv = time.time()

    print(g.T)

    print("Total time: ", end_pseudoinv-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", end_diff-end_simulation)
    print("Pseudoinv time: ", end_pseudoinv-end_diff)

    draw(t_points, y_list)

if __name__ == "__main__":
    infer_dynamic()