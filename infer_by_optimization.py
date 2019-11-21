# Estimate the ODEs by optimization methods
import numpy as np
from scipy.optimize import minimize
from generator import generate_complete_polynomail

def get_coef(t_points, y_list, order, stepsize):
    final_A_mat = None
    final_b_mat = None
    L_t = len(t_points)
    L_y = y_list[0].shape[1]
    gene = generate_complete_polynomail(L_y,order)
    L_p = gene.shape[0]
    for k in range(0,len(y_list)):
        y_points = y_list[k]
        coef_matrix = np.ones((L_t, L_p), dtype=np.double)
        for i in range(0,L_t):
            for j in range(0,L_p):
                for l in range(0,L_y):
                   coef_matrix[i][j] = coef_matrix[i][j] * (y_points[i][l] ** gene[j][l])

        A_matrix = coef_matrix[:L_t-1]
        b_matrix = (y_points[1:] - y_points[:L_t-1])/stepsize

        if k == 0:
            final_A_mat = A_matrix
            final_b_mat = b_matrix
        else :
            final_A_mat = np.r_[final_A_mat,A_matrix]
            final_b_mat = np.r_[final_b_mat,b_matrix]
    
    return final_A_mat, final_b_mat

def lambda_two_modes(final_A_mat, final_b_mat):
    def two_modes(x):
        A_row = final_A_mat.shape[0]
        A_col = final_A_mat.shape[1]
        b_col = final_b_mat.shape[1]
        sum = 0.0
        for i in range(0, A_row):
            mode1_sum = 0.0
            mode2_sum = 0.0
            for j in range(0, b_col):
                mode1_sum = mode1_sum + (final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])**2
                mode2_sum = mode2_sum + (final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])**2
            sum = sum +min(mode1_sum, mode2_sum)
        return sum
    return two_modes

def lambda_two_modes2(final_A_mat, final_b_mat):
    def two_modes(X):
        A_row = final_A_mat.shape[0]
        A_col = final_A_mat.shape[1]
        b_col = final_b_mat.shape[1]
        x = np.zeros((2*A_col,b_col))
        for i in range(0,2*A_col):
            for j in range(0,b_col):
                x[i][j] = X[i][j]
        print(A_row,A_col)
        # print(x[:A_col][:b_col].shape[0],x[:A_col][:b_col].shape[1])
        print(final_b_mat.shape[0],b_col)
        mat_1 = np.matmul(final_A_mat,x[:A_col]) - final_b_mat
        mat_2 = final_A_mat.dot(x[A_col:2*A_col]) - final_b_mat
        sum = 0.0
        for i in range(0, A_row):
            mode1_sum = 0.0
            mode2_sum = 0.0
            for j in range(0, b_col):
                mode1_sum = mode1_sum + mat_1[i][j]**2
                mode2_sum = mode2_sum + mat_2[i][j]**2
            sum = sum + min(mode1_sum, mode2_sum)
        return sum
    return two_modes




def infer_optimization(x0, A, b):
    return minimize(lambda_two_modes(A,b), x0, method='nelder-mead', options={'maxiter':100000, 'maxfev':100000, 'xatol': 1e-8, 'disp': True})
