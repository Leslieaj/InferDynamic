import numpy as np
from scipy import linalg, linspace
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time

from dynamics import dydx1, dydx2

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
    for k in range(0,len(y0)):
        y_object = solve_ivp(ode_func, (t_start, t_end+stepsize), y0[k], t_eval = t_points, rtol=1e-7, atol=1e-9)
        y_points = y_object.y.T
        y_list.append(y_points)
    return t_points, y_list

def diff_method(t_points, y_list, order, stepsize, highorder_flag=True):
    """Using multi-step difference method (Adams5) to calculate the coefficiant matrix.
    """
    final_A_mat = None
    final_b_mat = None
    L = len(t_points)
    D = L - 4    #Adams5

    for k in range(0,len(y_list)):
        y_points = y_list[k]
        A_matrix = np.zeros((D*2, order+2), dtype=np.double)
        b_matrix = np.zeros(D*2, dtype=np.double)
        coef_matrix = np.zeros((L, order+2), dtype=np.double)
        if highorder_flag == False:
            coef_matrix = np.zeros((L, order+1), dtype=np.double)
            A_matrix = np.zeros((D*2, order+1), dtype=np.double)
        
        #F[i] = [y[i]**4, y[i]**3, y[i]**2, y[i], 1]
        if highorder_flag == True:
            for i in range(0, L):
                for j in range(1, order+2):
                    coef_temp = y_points[i]**j
                    coef_matrix[i][order+1-j] = coef_temp
                coef_matrix[i][order+1] = 1
        else:
            for i in range(0, L):
                for j in range(1, order+1):
                    coef_temp = y_points[i]**j
                    coef_matrix[i][order-j] = coef_temp
                coef_matrix[i][order] = 1
        
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


# def dydx1(t,y):
#     dy_dx = -1.34*y**3+2.7*y**2-4*y+5.6
#     return dy_dx

# def dydx2(t,y):
#     dy_dx = -1.34*y**3+9.8*y**2+6.5*y-23
#     return dy_dx

def infer_dynamic():
    """ The main function to infer a dynamic system.
    """
    y0 = [[5],[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,4)
    stepsize = 0.02

    start = time.time()
    t_points, y_list = simulation_ode(dydx2, y0, t_tuple, stepsize)
    end_simulation = time.time()
    A, b = diff_method(t_points, y_list, 3, stepsize, highorder_flag=True)
    end_diff = time.time()
    g = linalg.pinv(A).dot(b.T)
    end_pseudoinv = time.time()

    print(g)

    print("Total time: ", end_pseudoinv-start)
    print("Simulation time: ", end_simulation-start)
    print("Calc-diff time: ", end_diff-end_simulation)
    print("Pseudoinv time: ", end_pseudoinv-end_diff)

    draw(t_points, y_list)

if __name__ == "__main__":
    infer_dynamic()
