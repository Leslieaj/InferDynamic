# Estimate the ODEs by optimization methods
import numpy as np
import time
from scipy.optimize import minimize, dual_annealing
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

        A_matrix = (coef_matrix[:L_t-1] + coef_matrix[1:])/2
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
        x1, x2 = np.array_split(x,2)
        x1 = np.mat(x1.reshape([A_col,b_col],order='F'))
        y1 = np.matmul(final_A_mat,x1) - final_b_mat
        y1 = np.multiply(y1,y1)
        y1 = y1.sum(axis=1)
        x2 = np.mat(x2.reshape([A_col,b_col],order='F'))
        y2 = np.matmul(final_A_mat,x2) - final_b_mat
        y2 = np.multiply(y2,y2)
        y2 = y2.sum(axis=1)
        go = np.minimum(y1,y2)
        go = go.sum()
        return go

        # for i in range(0, A_row):
        #     mode1_sum = 0.0
        #     mode2_sum = 0.0
        #     for j in range(0, b_col):
        #         mode1_sum = mode1_sum + (final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])**2
        #         mode2_sum = mode2_sum + (final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])**2
        #     sum = sum +min(mode1_sum, mode2_sum)
        # return sum
    return two_modes

def lambda_three_modes(final_A_mat, final_b_mat):
    def three_modes(x):
        A_row = final_A_mat.shape[0]
        A_col = final_A_mat.shape[1]
        b_col = final_b_mat.shape[1]
        x1, x2, x3 = np.array_split(x,3)
        x1 = np.mat(x1.reshape([A_col,b_col],order='F'))
        y1 = np.matmul(final_A_mat,x1) - final_b_mat
        y1 = np.multiply(y1,y1)
        y1 = y1.sum(axis=1)
        x2 = np.mat(x2.reshape([A_col,b_col],order='F'))
        y2 = np.matmul(final_A_mat,x2) - final_b_mat
        y2 = np.multiply(y2,y2)
        y2 = y2.sum(axis=1)
        x3 = np.mat(x3.reshape([A_col,b_col],order='F'))
        y3 = np.matmul(final_A_mat,x3) - final_b_mat
        y3 = np.multiply(y3,y3)
        y3 = y3.sum(axis=1)
        go = np.minimum(y1,y2)
        go = np.minimum(go,y3)
        go = go.sum()
        return go

        # for i in range(0, A_row):
        #     mode1_sum = 0.0
        #     mode2_sum = 0.0
        #     for j in range(0, b_col):
        #         mode1_sum = mode1_sum + (final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])**2
        #         mode2_sum = mode2_sum + (final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])**2
        #     sum = sum +min(mode1_sum, mode2_sum)
        # return sum
    return three_modes


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

def lambda_two_modes3(final_A_mat, final_b_mat):
    def two_modes(x):
        A_row = final_A_mat.shape[0]
        A_col = final_A_mat.shape[1]
        b_col = final_b_mat.shape[1]
        alpha = 5.0
        sum = 0.0
        for i in range(0, A_row):
            mode1_sum = 0.0
            mode2_sum = 0.0
            for j in range(0, b_col):
                mode1_sum = mode1_sum + (final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])**2
                mode2_sum = mode2_sum + (final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])**2
            # print("mode1_sum", mode1_sum)
            # print("mode2_sum", mode2_sum)
            # print("min", -np.log(np.exp(-alpha*mode1_sum)+np.exp(-alpha*mode2_sum))/alpha)
            sum = sum - np.log(np.exp(-alpha*mode1_sum)+np.exp(-alpha*mode2_sum))/alpha
            # print("sum", sum)
        return sum
    return two_modes

def lambda_two_modes3_der(final_A_mat, final_b_mat):
    def two_modes(x):
        A_row = final_A_mat.shape[0]
        A_col = final_A_mat.shape[1]
        b_col = final_b_mat.shape[1]
        der = np.zeros((1,2*b_col*A_col))
        mode1_sum = np.zeros((1,A_row))
        mode2_sum = np.zeros((1,A_row))
        for i in range(0, A_row):
            for j in range(0, b_col):
                mode1_sum[0][i] = mode1_sum[0][i] + (final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])**2
                mode2_sum[0][i] = mode2_sum[0][i] + (final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])**2
        
        for j in range(0,b_col):
            for k in range(0,A_col):
                for i in range(0,A_row):
                    der[0][j*A_col + k] = der[0][j*A_col + k] + np.exp(-mode1_sum[0][i])*final_A_mat[i][k]\
                        *2*(final_A_mat[i].dot(x[j*A_col:(j+1)*A_col])-final_b_mat[i][j])/(np.exp(-mode1_sum[0][i])+np.exp(-mode2_sum[0][i]))
        for j in range(0,b_col):
            for k in range(0,A_col):
                for i in range(0,A_row):
                    der[0][b_col*A_col + j*A_col + k] = der[0][b_col*A_col + j*A_col + k] + np.exp(-mode2_sum[0][i])*final_A_mat[i][k]\
                        *2*(final_A_mat[i].dot(x[b_col*A_col + j*A_col : b_col*A_col + (j+1)*A_col])-final_b_mat[i][j])/(np.exp(-mode1_sum[0][i])+np.exp(-mode2_sum[0][i]))
        return der
    return two_modes

def infer_optimization(x0, A, b):
    # print('result is:', lambda_two_modes3(A, b)(x0))
    # return minimize(lambda_two_modes(A,b), x0, method='nelder-mead', options={'maxiter':100000, 'maxfev':100000, 'xatol': 1e-8, 'disp': True})
    # return minimize(lambda_two_modes(A,b), x0, method='BFGS', jac=None, options={'maxiter':100000, 'gtol': 1e-05, 'disp': True})
    return minimize(lambda_two_modes(A,b), x0, method='CG',options={'maxiter':100000})
    # return dual_annealing(lambda_two_modes(A,b), bounds=[(-5,5)]*(2*A.shape[1]*b.shape[1]), maxfun=1000000, maxiter=100000)

def infer_optimization3(x0, A, b):
    # print('result is:', lambda_three_modes3(A, b)(x0))
    # return minimize(lambda_three_modes(A,b), x0, method='nelder-mead', options={'maxiter':100000, 'maxfev':100000, 'xatol': 1e-8, 'disp': True})
    # return minimize(lambda_three_modes(A,b), x0, method='BFGS', jac=None, options={'maxiter':100000, 'gtol': 1e-05, 'disp': True})
    return minimize(lambda_three_modes(A,b), x0, method='CG',options={'maxiter':100000})
    # return dual_annealing(lambda_three_modes(A,b), bounds=[(-5,5)]*(3*A.shape[1]*b.shape[1]), maxfun=1000000, maxiter=100000)
