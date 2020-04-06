import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
from dynamics import dydx3, fvdp2_1, fvdp3_1, mode2_1, mode2_11, mode2_1_test, conti_test, conti_test_test, conti_test1, ode_test
from infer_multi_ch import simulation_ode, simulation_ode_stiff, infer_dynamic, parti, infer_dynamic_modes_ex, infer_dynamic_modes_exx, dist, diff_method, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie

import dynamics
import warnings
warnings.filterwarnings('ignore')

import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *



def case1():
    y0 = [[1,7]]
    t_tuple = [(0,25)]
    stepsize = 0.01
    order = 2
    maxorder = 4

    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.05)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0],tpar_list[i][-1])
        print(ypar_list[i][0],ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic([tpar_list[0],tpar_list[1]], [ypar_list[0],ypar_list[1]], stepsize, 4)
    # print(result_coef)
    # tstart = time.time()
    # comt,comy = simulation_ode_stiff(ode_test(result_coef,4), [ypar_list[0][0],ypar_list[1][0]], [(tpar_list[0][0], tpar_list[0][-1]),(tpar_list[1][0], tpar_list[1][-1])], stepsize,eps=0)
    # tend = time.time()
    # print(dist(comy,[ypar_list[0],ypar_list[1]]))
    # print(tend-tstart)
    # print(dist([ypar_list[0],ypar_list[2],ypar_list[4]],comy))
    # result_coef = np.matrix([[ 0, 0, 0, -0.26, 0.26, 0], [0, 0, 0, 0, 0, -1]])
    # comt,comy = simulation_ode(ode_test(result_coef,2), [ypar_list[0][0],ypar_list[2][0],ypar_list[4][0]], [(tpar_list[0][0], tpar_list[0][-1]),(tpar_list[2][0], tpar_list[2][-1]),(tpar_list[4][0], tpar_list[4][-1])], stepsize,eps=0)
    # print(dist([ypar_list[0],ypar_list[2],ypar_list[4]],comy))
    # result_coef = np.matrix([[ 0, 0, 0, -0.26, 0.26, 0], [0, 0, 0, 0, 0, 1]])
    # comt,comy = simulation_ode(ode_test(result_coef,2), [ypar_list[1][0],ypar_list[3][0],ypar_list[5][0]], [(tpar_list[1][0], tpar_list[1][-1]),(tpar_list[3][0], tpar_list[3][-1]),(tpar_list[5][0], tpar_list[5][-1])], stepsize,eps=0)
    # print(dist([ypar_list[1],ypar_list[3],ypar_list[5]],comy))
    tstart = time.time()
    modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.001)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    tend = time.time()
    print(modes)
    print(coefs)
    print(mdors)
    print(tend-tstart)


    
def case2():
    # y0 = [[1,3],[-1,-2],[-3,-5],[2,4],[-2,3],[-4,5],[4,7],[-2,-10],[5,8]]
    # t_tuple = [(0,2),(0,3),(0,4),(0,5),(0,1),(0,1),(0,5),(0,6),(0,6)]
    y0 = [[1,3]]
    t_tuple = [(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 4
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    # modes, coefs = infer_dynamic_modes(t_list, y_list, stepsize, maxorder, 0.001)
    modes, coefs, mdors = infer_dynamic_modes_ex(t_list, y_list, stepsize, maxorder, 0.01)
    print(modes)
    print(coefs)
    print(mdors)
    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()



def case3():
    y0 = [[9]]
    t_tuple = [(0,3)]
    stepsize = 0.01
    order = 2
    t_list, y_list = simulation_ode(dydx3, y0, t_tuple, stepsize, eps=0)
    for i in range(0,len(y_list)):
        plt.plot(t_list[i],y_list[i])
    plt.show()


def case4():
    y0 = [[0,0],[1,0]]
    t_tuple = [(0,2.5),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 4

    # start = time.time()
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.0001)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.0001)
    print(modes)
    print(coefs)
    print(mdors)

def case5():
    y0 = [[0,0],[1,0]]
    t_tuple = [(0,2.5),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 3

    
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)

    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # print(modes)
    # print(coefs)
    # print(mdors)
    ttest_list=[]
    ttest_list.append(tpar_list[0])
    ttest_list.append(tpar_list[1])
    ytest_list=[]
    ytest_list.append(ypar_list[0])
    ytest_list.append(ypar_list[1])
    A, b = diff_method(ttest_list, ytest_list, 3, stepsize)
    g = pinv2(A).dot(b)
    print(g.T)
    t_start = tpar_list[0][0]
    t_end = tpar_list[0][-1]
    t_start = 0
    t_end = 0.01
    t_points = np.arange(t_start, t_end + stepsize, stepsize)
    tstart = time.time()
    y_object = solve_ivp(ode_test(g.T,3), (t_start, t_end+stepsize), ypar_list[0][0], t_eval = t_points, rtol=1e-7, atol=1e-9)
    y_points = y_object.y.T
    tend = time.time()
    print(y_points)
    print(tend-tstart)



def case6():
    y0 = [[1,7]]
    t_tuple = [(0,30)]
    stepsize = 0.001
    order = 2
    maxorder = 4
    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0],tpar_list[i][-1])
        print(ypar_list[i][0],ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic([tpar_list[0],tpar_list[1]], [ypar_list[0],ypar_list[1]], stepsize, 4)
    # print(result_coef)
    # tstart = time.time()
    # comt,comy = simulation_ode_stiff(ode_test(result_coef,4), [ypar_list[0][0],ypar_list[1][0]], [(tpar_list[0][0], tpar_list[0][-1]),(tpar_list[1][0], tpar_list[1][-1])], stepsize,eps=0)
    # tend = time.time()
    # print(dist(comy,[ypar_list[0],ypar_list[1]]))
    # print(tend-tstart)
    # print(dist([ypar_list[0],ypar_list[2],ypar_list[4]],comy))
    # result_coef = np.matrix([[ 0, 0, 0, -0.26, 0.26, 0], [0, 0, 0, 0, 0, -1]])
    # comt,comy = simulation_ode(ode_test(result_coef,2), [ypar_list[0][0],ypar_list[2][0],ypar_list[4][0]], [(tpar_list[0][0], tpar_list[0][-1]),(tpar_list[2][0], tpar_list[2][-1]),(tpar_list[4][0], tpar_list[4][-1])], stepsize,eps=0)
    # print(dist([ypar_list[0],ypar_list[2],ypar_list[4]],comy))
    # result_coef = np.matrix([[ 0, 0, 0, -0.26, 0.26, 0], [0, 0, 0, 0, 0, 1]])
    # comt,comy = simulation_ode(ode_test(result_coef,2), [ypar_list[1][0],ypar_list[3][0],ypar_list[5][0]], [(tpar_list[1][0], tpar_list[1][-1]),(tpar_list[3][0], tpar_list[3][-1]),(tpar_list[5][0], tpar_list[5][-1])], stepsize,eps=0)
    # print(dist([ypar_list[1],ypar_list[3],ypar_list[5]],comy))
    tstart = time.time()
    labels = infer_dynamic_modes_ex_dbs(tpar_list, ypar_list, stepsize, maxorder, 0.001)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    tend = time.time()
    print(labels)


def case7():
    y0 = [[0,0],[1,0],[0.3,0],[2.7,0]]
    t_tuple = [(0,2.5),(0,2),(0,2),(0,1)]
    stepsize = 0.01
    order = 2
    maxorder = 4

    # start = time.time()
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    labels = infer_dynamic_modes_ex_dbs(tpar_list, ypar_list, stepsize, maxorder, 0.0001)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.0001)
    print(labels)





def case8():
    # y0 = [[1,7]]
    # t_tuple = [(0,30)]
    # stepsize = 0.001
    # order = 2
    # maxorder = 4
    # t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0.00001)


    y0 = [[0,0],[1,0],[0.3,0]]
    t_tuple = [(0,2.5),(0,2),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 4
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
        for i in range(0,temp_y.shape[0]-4):
            g = temp_y[i][:] - 4*temp_y[i+1][:] + 6*temp_y[i+2][:] - 4*temp_y[i+3][:] + temp_y[i+4][:]
            gg = np.max(g)
            if gg > 0.005:
                plt.scatter(temp_y[i+4][0], temp_y[i+4][1], s= 10, c='r')
    plt.show()
    
    # tstart = time.time()
    # labels = infer_dynamic_modes_ex_dbs(tpar_list, ypar_list, stepsize, maxorder, 0.001)
    # modes, coefs, mdors = infer_dynamic_modes_exx(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # tend = time.time()
    # print(labels)


def case9():
    y0 = [[0,0]]
    t_tuple = [(0,7)]
    stepsize = 0.001
    order = 2
    maxorder = 4
    t_list, y_list = simulation_ode(mode2_11, y0, t_tuple, stepsize, eps=0)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])
    labels = infer_dynamic_modes_ex_dbs(tpar_list, ypar_list, stepsize, maxorder, 0.0001)
    print(labels)



def case10():
    y0 = [[0,0],[1,0],[0.3,0],[2.7,0]]
    t_tuple = [(0,2.5),(0,2),(0,2),(0,1)]
    stepsize = 0.01
    order = 2
    maxorder = 3

    # start = time.time()
    t_list, y_list = simulation_ode(conti_test, y0, t_tuple, stepsize, eps=0)
    # end_simulation = time.time()
    # result_coef, calcdiff_time, pseudoinv_time = infer_dynamic(t_list, y_list, stepsize, order)
    # end_inference = time.time()

    # print(result_coef)
    # print()
    # print("Total time: ", end_inference-start)
    # print("Simulation time: ", end_simulation-start)
    # print("Calc-diff time: ", calcdiff_time)
    # print("Pseudoinv time: ", pseudoinv_time)

    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    G,labels = infer_dynamic_modes_pie(tpar_list, ypar_list, stepsize, maxorder, 0.15)
    print(labels)
    print(G)


def case11():
    y0 = [[1,7],[2,6]]
    t_tuple = [(0,2),(0,2)]
    stepsize = 0.01
    order = 2
    maxorder = 2
    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(len(tpar_list))
    for i in range(0,len(tpar_list)):
        print(tpar_list[i][0])
        print(tpar_list[i][-1])
        print(ypar_list[i][0])
        print(ypar_list[i][-1])
    G,labels = infer_dynamic_modes_pie(tpar_list, ypar_list, stepsize, maxorder, 0.02)
    # modes, coefs, mdors = infer_dynamic_modes_ex(tpar_list, ypar_list, stepsize, maxorder, 0.01)
    # print(modes)
    # print(coefs)
    # print(mdors)
    print(len(labels))
    print(G)


if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    # case4()
    # case5()
    # case6()
    # case7()
    # case8()
    # case9()
    # case10()
    case11()
    
