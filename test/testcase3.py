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
from dynamics import mode2_1, conti_test1
from infer_single import simulation_ode
from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *


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

    # pr = cProfile.Profile()
    # pr.enable()
    results = infer_optimization(x0, final_A_mat, final_b_mat)
    # p = Stats(pr)
    # p.strip_dirs()
    # p.sort_stats('cumtime')
    # p.print_stats(100)
    
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

    start_svm = time.time()
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
            modet[i] = -1
    # plt.plot(t_points,y_list[0])   
    # plt.plot(t_points[:A_row],modet)
    # plt.show() 
    y = []
    x = []
    for i in range(0,A_row):
        y.append(modet[i])
        x_0 = y_list[0][i][0]
        x_1 = y_list[0][i][1]
        x.append({1:x_0, 2:x_1})
    
    prob  = svm_problem(y, x)
    param = svm_parameter('-t 0 -c 200 -b 1')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    end_svm_training = time.time()

    nsv = m.get_nr_sv()
    y0cof = 0.0
    y1cof = 0.0
    for i in range(0,nsv):
        if m.get_SV()[i].__contains__(1):
            y0cof = y0cof + m.get_sv_coef()[i][0]*m.get_SV()[i][1]
        if m.get_SV()[i].__contains__(2):
            y1cof = y1cof + m.get_sv_coef()[i][0]*m.get_SV()[i][2]
    print("y0 coef: ", y0cof)
    print("y1 coef: ", y1cof)
    print("constant: ", -m.rho[0])

    mode1_y0 = []
    mode1_y1 = []
    mode2_y0 = []
    mode2_y1 = []
    for i in range(0,A_row):
        if modet[i] > 0 :
            mode1_y0.append(y_list[0][i][0])
            mode1_y1.append(y_list[0][i][1])
        else:
            mode2_y0.append(y_list[0][i][0])
            mode2_y1.append(y_list[0][i][1])
    p_label, p_acc, p_val = svm_predict(y,x,m)
    end_svm_predict = time.time()
    print(p_acc)
    print("SVM training time: ", end_svm_training-start_svm)
    print("SVM predict time: ", end_svm_predict-end_svm_training)
    print("total time: ", end_svm_predict-start)
    plt.scatter(mode1_y0,mode1_y1,c='r',s=0.1)
    plt.scatter(mode2_y0,mode2_y1,c='g',s=0.1)
    yy1 = np.arange(-3,3,0.001)
    yy0 = (m.rho[0] - y1cof*yy1)/y0cof
    plt.plot(yy0,yy1)
    plt.show()
    

def case2():
    y0 = [[0.1,0],[0.4,0],[0.33,0]]
    t_tuple = (0,1)
    stepsize = 0.001
    order = 2

    start = time.time()
    t_points, y_list = simulation_ode(conti_test1, y0, t_tuple, stepsize, eps=0)
    end_simulation = time.time()

    final_A_mat, final_b_mat = get_coef(t_points, y_list, order, stepsize)
    end_coedf = time.time()

    # x0 = np.ones((2*final_A_mat.shape[1], final_b_mat.shape[1]))*0.1
    # x0 = np.ones((1,2*final_A_mat.shape[1]*final_b_mat.shape[1]))*0.1
    print(2*final_A_mat.shape[1]*final_b_mat.shape[1])
    x0 = np.random.uniform(-5,5,[2*final_A_mat.shape[1]*final_b_mat.shape[1]])
    # print(x0)
    x1 = np.array([0,0,0,-1,0,4,0,0,0, 0,0,1.0, -1,0,0,2.5,0,1, 0,0,0, 0,0,1.0])
    x2 = np.array([0,0,0,-1.5,0,5,0,0,0, 0,0,1.0, -1,0,0,2.5,0,1, 0,0,0, 0,0,1.0])

    # pr = cProfile.Profile()
    # pr.enable()
    results = infer_optimization(x0, final_A_mat, final_b_mat)
    # p = Stats(pr)
    # p.strip_dirs()
    # p.sort_stats('cumtime')
    # p.print_stats(100)
    
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

    # start_svm = time.time()
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
            modet[i] = -1
    modett = np.array_split(modet,len(y0))
    print(A_row)
    for i in range(0,len(y0)):
        plt.plot(t_points,y_list[i])   
        plt.plot(t_points[:len(modett[i])],modett[i])
    # plt.plot(t_points,y_list[2])   
    # plt.plot(t_points[:len(modett[0])],modett[2])
    plt.show()   

if __name__ == "__main__":
    # case1()
    case2()