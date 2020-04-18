import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
# from scipy import linalg, linspace
# from scipy.integrate import odeint, solve_ivp
import time
import sys, os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir
from sklearn import linear_model
from sklearn.linear_model import Ridge

from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
from dynamics import dydx3, fvdp2_1, fvdp3_1, fvdp3_2, fvdp3_3, \
    mode2_1, mode2_11, mode2_1_test, \
    conti_test, conti_test_test, conti_test1, ode_test, \
    mode1, mode2, event2, \
    mmode1, mmode2, event3, mmode, \
    incubator_mode1, incubator_mode2, event1, \
    modetr_1, modetr_2, modetr_3, eventtr_1, eventtr_2
from infer_multi_ch import simulation_ode, infer_dynamic, parti, infer_dynamic_modes_ex, norm, reclass, dropclass, \
    infer_dynamic_modes_exx, dist, diff_method, diff_method1, infer_dynamic_modes_ex_dbs, infer_dynamic_modes_pie, \
    infer_dynamic_modes_new, diff_method_new1, diff_method_new, simulation_ode_2, simulation_ode_3, diff_method_backandfor

import infer_multi_ch
from generator import generate_complete_polynomial
import dynamics
import warnings
warnings.filterwarnings('ignore')

import math
import cProfile
from pstats import Stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ADD the path of the parent dir

from draw import draw, draw2D, draw2D_dots, draw3D
from infer_by_optimization import lambda_two_modes, get_coef, infer_optimization, lambda_two_modes3, infer_optimization3, lambda_three_modes
# from libsvm.svm import svm_problem, svm_parameter
# from libsvm.svmutil import svm_train, svm_predict
from libsvm.svmutil import *




# def final_infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep=0.1):
#     A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
#     P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, ep)
#     for i in range(0,len(P)):
        



def case1():
    # y0 = [[1,3],[-1,-2],[-3,-5],[2,4]]
    # t_tuple = [(0,20),(0,10),(0,15),(0,15)]
    y0 = [[1,3],[-1,-2]]
    t_tuple = [(0,20),(0,10)]
    stepsize = 0.01
    order = 2
    maxorder = 2
    # start = time.time()
    t_list, y_list = simulation_ode(mode2_1, y0, t_tuple, stepsize, eps=0)

    # for temp_y in y_list:
    #     y0_list = temp_y.T[0]
    #     y1_list = temp_y.T[1]
    #     plt.plot(y0_list,y1_list,'b')
    # plt.show()
    
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    
    print(P)
    print(G)
    print(D)
    y = []
    x = []

    for j in range(0,len(P[0])):
        y.append(1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})
    
    for j in range(0,len(P[1])):
        y.append(-1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 1 -c 10 -b 0 ')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    yt = [-1,-1,1,1]
    xt = [{1:1, 2:1},{1:-1, 2:1},{1:1, 2:-1},{1:-1, 2:-1}]
    print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # print(p_label)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    
    # def clafun(x):
    #     g = -m.rho[0]
    #     for i in range(0,nsv):
    #         g = g + svc[i][0] * ((0.5 * (x[0]*sv[i][1] + x[1]*sv[i][2]))**3)
    #     return g

    g = -m.rho[0]
    a1 = 0
    a2 = 0
    for i in range(0,nsv):
        a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
        a2 = a2 + svc[i][0] * 0.5 * sv[i][2]

    print("a1",a1/a1)
    print("a2",a2/a1)
    print("g",g/a1)
    
    
    
 
    # ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    # plt.show()
    dim = G[0].shape[0]
    A = generate_complete_polynomial(dim,maxorder)
    def odepre(t,y):
        # print("in")
        basicf = []
        for i in range(0,A.shape[0]):
            ap = 1
            for j in range(0,A.shape[1]):
                ap = ap*(y[j]**A[i][j])
            basicf.append(ap)
        b = np.array(basicf)
        dydt = np.zeros(dim)
        if a1 * y[0] + a2 * y[1] + g > 0: 
            for l in range(0,dim):
                dydt[l] = G[0][l].dot(b)
        else:
            for l in range(0,dim):
                dydt[l] = G[1][l].dot(b)
        # print("out")
        return dydt
    py0 = [[2,4],[-1,-3]]
    pt_tuple = [(0,10),(0,10)]
    start = time.time()
    print("origin")
    tp_list, yp_list = simulation_ode(mode2_1, py0, pt_tuple, stepsize, eps=0)
    end1 = time.time()
    print("predict")
    tpre_list, ypre_list = simulation_ode(odepre, py0, pt_tuple, stepsize, eps=0)
    end2 = time.time()
    print("simutime",end1-start)
    print("predtime",end2-end1)

    for temp_y in yp_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    for temp_y in ypre_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'r')
    plt.show()



def case2():
    y0 = [[0,0],[1,0],[0.3,0],[2.7,0]]
    t_tuple = [(0,2.5),(0,2),(0,2),(0,1)]
    stepsize = 0.001
    order = 2
    maxorder = 2

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

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    print(P)
    print(G)
    print(D)



def case3():
    y0 = [[5,5,5],[2,2,2]]
    t_tuple = [(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    t_list, y_list = simulation_ode(fvdp3_3, y0, t_tuple, stepsize, eps=0)
    draw3D(y_list)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    # A, b1, b2, Y = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    # print(b1)
    # print(b2)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.02)
    print(P)
    print(G)
    print(D)
    # y = []
    # x = []

    # for j in range(0,len(P[0])):
    #     y.append(1)
    #     x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1], 3:Y[P[0][j],2]})
    
    # for j in range(0,len(P[1])):
    #     y.append(-1)
    #     x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1], 3:Y[P[0][j],2]})

    # prob  = svm_problem(y, x)
    # param = svm_parameter('-t 1 -d 1 -c 10 -b 0 ')
    # m = svm_train(prob, param)
    # svm_save_model('model_file', m)
    # print("pred")
    # p_label, p_acc, p_val = svm_predict(y, x, m)
    # # print(p_label)
    # nsv = m.get_nr_sv()
    # svc = m.get_sv_coef()
    # sv = m.get_SV()
    # g = -m.rho[0]
    # a1 = 0
    # a2 = 0
    # a3 = 0
    # for i in range(0,nsv):
    #     a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
    #     a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
    #     a2 = a2 + svc[i][0] * 0.5 * sv[i][3]

    # print("a1",a1)
    # print("a2",a2)
    # print("a3",a3)
    # print("g",g)
    

def case4():
    y0 = [[5,5,5],[2,2,2]]
    t_tuple = [(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    t_list, y_list = simulation_ode(fvdp3_3, y0, t_tuple, stepsize, eps=0)
    draw3D(y_list)
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    G,labels = infer_dynamic_modes_pie(tpar_list, ypar_list, stepsize, maxorder, 0.02)
    print(labels)
    print(G)
    # A, b, Y = diff_method_new1(t_list, y_list, maxorder, stepsize)
    # P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.03)
    # print(P)
    # print(G)
    # print(D)
    # y = []
    # x = []

    # for j in range(0,len(P[0])):
    #     y.append(1)
    #     x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1], 3:Y[P[0][j],2]})
    
    # for j in range(0,len(P[1])):
    #     y.append(-1)
    #     x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1], 3:Y[P[0][j],2]})

    # prob  = svm_problem(y, x)
    # param = svm_parameter('-t 1 -d 1 -c 10 -b 0 ')
    # m = svm_train(prob, param)
    # svm_save_model('model_file', m)
    # print("pred")
    # p_label, p_acc, p_val = svm_predict(y, x, m)
    # # print(p_label)
    # nsv = m.get_nr_sv()
    # svc = m.get_sv_coef()
    # sv = m.get_SV()
    # g = -m.rho[0]
    # a1 = 0
    # a2 = 0
    # a3 = 0
    # for i in range(0,nsv):
    #     a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
    #     a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
    #     a2 = a2 + svc[i][0] * 0.5 * sv[i][3]

    # print("a1",a1)
    # print("a2",a2)
    # print("a3",a3)
    # print("g",g)


def case5():
    y0 = [[5,5,5], [2,2,2]]
    t_tuple = [(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    t_list, y_list = simulation_ode(fvdp3_3, y0, t_tuple, stepsize, eps=0)
    clfs, boundary = infer_multi_ch.infer_multi_linear(t_list, y_list, stepsize, maxorder)
    for clf in clfs:
        print(clf.coef_)
    print(boundary)
    num_pt0 = y_list[0].shape[0]
    num_pt1 = y_list[1].shape[0]
    sum = 0
    for i in range(100):

        # index = random.choice(range(num_pt0))
        x = y_list[0][i]
        # sum += infer_multi_ch.test_classify(fvdp3_3, clfs, boundary, maxorder, x)
        print(infer_multi_ch.test_classify(fvdp3_3, clfs, boundary, maxorder, x))
    # for i in range(num_pt1):

    #     # index = random.choice(range(num_pt0))
    #     x = y_list[1][i]
    #     sum += infer_multi_ch.test_classify(fvdp3_3, clfs, boundary, maxorder, x)
    
    # print(sum/(num_pt0+num_pt1))


def case6():
    y0 = [[0,1],[0,2],[0,3],[0,4],[0,5]]
    t_tuple = [(0,20),(0,20),(0,20),(0,20),(0,20)]
    stepsize = 0.01
    maxorder = 3
    # start = time.time()
    t_list, y_list = simulation_ode_2([mode1, mode2], event2, y0, t_tuple, stepsize)

    for temp_y in y_list:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list,'b')
    plt.show()
    # clfs, boundary = infer_multi_ch.infer_multi_linear_new(t_list, y_list, stepsize, maxorder)
    # for clf in clfs:
    #     print('g',clf.coef_)
    # print(boundary)
    A, b, Y = diff_method_new1(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    print(P)
    print(G)
    print(D)
    y = []
    x = []

    for j in range(0,len(P[0])):
        y.append(1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})
    
    
    for j in range(0,len(P[1])):
        y.append(-1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})


    y0_list = []
    y1_list = []
    for j in range(0,len(P[0])):
        y0_list.append(Y[P[0][j],0])
        y1_list.append(Y[P[0][j],1])
    
    plt.scatter(y0_list,y1_list,s=1)
    y0_list = []
    y1_list = []
    
    for j in range(0,len(P[1])):
        y0_list.append(Y[P[1][j],0])
        y1_list.append(Y[P[1][j],1])
    
    plt.scatter(y0_list,y1_list,s=1)
    

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 2 -r 1 -c 10 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # # print(p_label)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    
    def f(x,y):
        g = -m.rho[0]
        for i in range(0,nsv):
            g = g + svc[i][0] * (0.5*(x*sv[i][1]+y*sv[i][2])+1)**2
        return g

    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    
    X,Y = np.meshgrid(x,y)#将x，y指传入网格中
    # plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)#8指图中的8+1根线，绘制等温线，其中cmap指颜色
    
    C = plt.contour(X,Y,f(X,Y),[0])#colors指等高线颜色
    plt.clabel(C,inline=True,fontsize=10)#inline=True指字体在等高线中
    
    plt.xticks(())
    plt.yticks(())
    plt.show()
    



def case7():
    y0 = [[4,0.1,3.1,0],[5.9,0.2,-3,0],[4.1,0.5,2,0],[6,0.7,2,0]]
    t_tuple = [(0,5),(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    # start = time.time()
    t_list, y_list = simulation_ode_2([mmode1, mmode2], event3, y0, t_tuple, stepsize)

    # for temp_y in y_list:
    #     y0_list = temp_y.T[0]
    #     y1_list = temp_y.T[1]
    #     plt.plot(y0_list,y1_list,'b')
    # plt.show()
    A, b, Y = diff_method_new1(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.03)
    print(P)
    print(G)
    print(D)
    y = []
    x = []

    for j in range(0,len(P[0])):
        y.append(1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1], 3:Y[P[0][j],2], 4:Y[P[0][j],3]})
    
    for j in range(0,len(P[1])):
        y.append(-1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1], 3:Y[P[1][j],2], 4:Y[P[1][j],3]})

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 1 -c 10 -r 1 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    # print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # print(p_label)
    print('pre',p_acc[0])
    # print(p_val)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    
    # def clafun(x):
    #     g = -m.rho[0]
    #     for i in range(0,nsv):
    #         g = g + svc[i][0] * ((0.5 * (x[0]*sv[i][1] + x[1]*sv[i][2]))**3)
    #     return g

    g = -m.rho[0]
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for i in range(0,nsv):
        a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
        a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
        a3 = a3 + svc[i][0] * 0.5 * sv[i][3]
        a4 = a4 + svc[i][0] * 0.5 * sv[i][4]
        g = g + svc[i][0]*1


    # print("a1",a1/a1)
    # print("a2",a2/a1)
    # print("g",g/a1)
    
    # def classify_mode(x):
    #     return a1 * x[0] + a2 * x[1] + a3 * x[2] + a4 * x[3] + g > 0
    #     # return a.dot(x.T) + g >0
    
    # def get_poly_pt(x):
    #     gene = generate_complete_polynomial(len(x), maxorder)
    #     val = []
    #     for i in range(gene.shape[0]):
    #         val.append(1.0)
    #         for j in range(gene.shape[1]):
    #             val[i] = val[i] * (x[j] ** gene[i,j])
    #     return val
    
    # def predict_deriv(x):
    #     poly_pt = np.mat(get_poly_pt(x))
    #     if classify_mode(x):
    #         return np.matmul(poly_pt,G[0].T)
    #     else:
    #         return np.matmul(poly_pt,G[1].T)

    # sum = 0
    # num = 0
    # for ytemp in y_list:
    #     num_pt0 = ytemp.shape[0]
    #     num = num + num_pt0
    #     for i in range(0,num_pt0):
    #     # index = random.choice(range(num_pt0))
    #         x = ytemp[i]
    #         diff1 = np.array(mmode(0.0, x))
    #         print('1',diff1)
    #         diffm = predict_deriv(x)
    #         diff2 = np.zeros(diffm.shape[1])
    #         for l in range(diffm.shape[1]):
    #             diff2[l] = diffm[0,l]
    #         print('2',diff2)
    #         err = norm(diff1-diff2)/(norm(diff1)+norm(diff2))
    #         sum = sum +err
    # print(sum/num)

def case8():
    y0 = [[4,0.1,3.1,0],[5.9,0.2,-3,0],[4.1,0.5,2,0],[6,0.7,2,0]]
    t_tuple = [(0,5),(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    # start = time.time()
    t_list, y_list = simulation_ode_2([mmode1, mmode2], event3, y0, t_tuple, stepsize)
    clfs, boundary = infer_multi_ch.infer_multi_linear_new(t_list, y_list, stepsize, maxorder)
    for clf in clfs:
        print(clf.coef_)
    print(boundary)
    num_pt0 = y_list[0].shape[0]
    num_pt1 = y_list[1].shape[0]
    sum = 0
    for i in range(num_pt0):

        # index = random.choice(range(num_pt0))
        x = y_list[0][i]
        sum += infer_multi_ch.test_classify(mmode, clfs, boundary, maxorder, x)
        # print(infer_multi_ch.test_classify(fvdp3_3, clfs, boundary, maxorder, x))
    for i in range(num_pt1):

        # index = random.choice(range(num_pt0))
        x = y_list[1][i]
        sum += infer_multi_ch.test_classify(mmode, clfs, boundary, maxorder, x)
    
    print(sum/(num_pt0+num_pt1))



def case9():
    y0 = [[4.7,0.2,-6,0,0,3]]
    t_tuple = [(0,10)]
    stepsize = 0.001
    maxorder = 1
    # start = time.time()
    t_list, y_list = simulation_ode_2([mmode1, mmode2], event3, y0, t_tuple, stepsize)
    tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    print(tpar_list[0].shape)
    tt = [tpar_list[0],tpar_list[2],tpar_list[4]]
    yy = [ypar_list[0],ypar_list[2],ypar_list[4]]
    A ,b = diff_method1(tt, yy, maxorder, stepsize)
    # # clf = linear_model.LinearRegression(fit_intercept=False,normalize=True)
    # # clf.fit(A,b)
    # # g = clf.coef_
    # bb = clf.predict(A)
    g = pinv2(A).dot(b)

    print("g=",g.T)
    # print('b',b)
    # print('bb',bb)
    # G,labels = infer_dynamic_modes_pie(tpar_list, ypar_list, stepsize, maxorder, 0.02)
    # print(labels)
    # print(G)


def case10():
    y0 = [[-1,1],[1,4],[2,-3]]
    t_tuple = [(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    # start = time.time()
    def labeltest(y):
        if eventtr_1(0,y)<0 and eventtr_2(0,y)>0:
            return 0
        elif eventtr_1(0,y)>=0 and eventtr_2(0,y)>0:
            return 1
        else:
            return 2

    t_list, y_list = simulation_ode_3([modetr_1, modetr_2, modetr_3], [eventtr_1,eventtr_2,eventtr_2], labeltest, y0, t_tuple, stepsize)
    draw2D(y_list)
    A, b, Y = diff_method_new(t_list, y_list, maxorder, stepsize)
    P,G,D = infer_dynamic_modes_new(t_list, y_list, stepsize, maxorder, 0.01)
    print(P)
    print(G)
    print(D)
    # for p in P:
    #     print(len(p))
    # tpar_list,ypar_list = parti(t_list,y_list,0.2,1/3)
    
    # for temp in tpar_list:
    #     print(temp[-1])
    # labels = infer_dynamic_modes_ex_dbs(tpar_list, ypar_list, stepsize, maxorder, 0.02)
    # print(labels)
    # draw2D(ypar_list)
    for i in range(0,len(P)):
        y0_list = []
        y1_list = []
        for j in range(0,len(P[i])):
            y0_list.append(Y[P[i][j],0])
            y1_list.append(Y[P[i][j],1])
    
        plt.scatter(y0_list,y1_list,s=1)
    plt.show()
    
    P,G = reclass(A,b,P,0.01)
    for p in P:
        print(len(p))
    print(G)

    for i in range(0,len(P)):
        y0_list = []
        y1_list = []
        for j in range(0,len(P[i])):
            y0_list.append(Y[P[i][j],0])
            y1_list.append(Y[P[i][j],1])
    
        plt.scatter(y0_list,y1_list,s=1)
    plt.show()
    P,D = dropclass(P,G,D,A,b,Y,0.01,0.01)
    print(D)
    for i in range(0,len(P)):
        y0_list = []
        y1_list = []
        for j in range(0,len(P[i])):
            y0_list.append(Y[P[i][j],0])
            y1_list.append(Y[P[i][j],1])
    
        plt.scatter(y0_list,y1_list,s=1)
    # plt.show()
    
    y=[]
    x=[]

    for j in range(0,len(P[2])):
        y.append(1)
        x.append({1:Y[P[2][j],0], 2:Y[P[2][j],1]})
    
    for j in range(0,len(P[1])):
        y.append(-1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})
    
    for j in range(0,len(P[0])):
        y.append(-1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 1 -c 100 -r 1 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # print(p_label)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    
    # def clafun(x):
    #     g = -m.rho[0]
    #     for i in range(0,nsv):
    #         g = g + svc[i][0] * ((0.5 * (x[0]*sv[i][1] + x[1]*sv[i][2]))**3)
    #     return g

    g = -m.rho[0]
    a1 = 0
    a2 = 0
    for i in range(0,nsv):
        a1 = a1 + svc[i][0] * 0.5 * sv[i][1]
        a2 = a2 + svc[i][0] * 0.5 * sv[i][2]
        g = g + svc[i][0]*1
    print(a1)
    print(a2)
    print(g)

    def f(x,y):
        g = -m.rho[0]
        for i in range(0,nsv):
            g = g + svc[i][0] * (0.5*(x*sv[i][1]+y*sv[i][2])+1)
        return g

    x=[]
    y=[]

    for j in range(0,len(P[1])):
        y.append(1)
        x.append({1:Y[P[1][j],0], 2:Y[P[1][j],1]})
    
    for j in range(0,len(P[0])):
        y.append(-1)
        x.append({1:Y[P[0][j],0], 2:Y[P[0][j],1]})

    prob  = svm_problem(y, x)
    param = svm_parameter('-t 1 -d 1 -c 100 -r 1 -b 0')
    m = svm_train(prob, param)
    svm_save_model('model_file', m)
    print("pred")
    p_label, p_acc, p_val = svm_predict(y, x, m)
    # print(p_label)
    nsv = m.get_nr_sv()
    svc = m.get_sv_coef()
    sv = m.get_SV()
    # print(nsv)
    # print(svc)
    # print(sv)
    
    # def clafun(x):
    #     g = -m.rho[0]
    #     for i in range(0,nsv):
    #         g = g + svc[i][0] * ((0.5 * (x[0]*sv[i][1] + x[1]*sv[i][2]))**3)
    #     return g

    g1 = -m.rho[0]
    b1 = 0
    b2 = 0
    for i in range(0,nsv):
        b1 = b1 + svc[i][0] * 0.5 * sv[i][1]
        b2 = b2 + svc[i][0] * 0.5 * sv[i][2]
        g1 = g1 + svc[i][0]*1
    # print(a1)
    # print(a2)
    # print(g)

    def h(x,y):
        g = -m.rho[0]
        for i in range(0,nsv):
            g = g + svc[i][0] * (0.5*(x*sv[i][1]+y*sv[i][2])+1)
        return g
    x = np.linspace(-10,10,100)
    y = np.linspace(-10,10,100)
    
    X,Y = np.meshgrid(x,y)#将x，y指传入网格中
    # plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)#8指图中的8+1根线，绘制等温线，其中cmap指颜色
    
    C = plt.contour(X,Y,f(X,Y),[0])#colors指等高线颜色
    plt.clabel(C,inline=True,fontsize=10)#inline=True指字体在等高线中
    D = plt.contour(X,Y,h(X,Y),[0])#colors指等高线颜色
    plt.clabel(D,inline=True,fontsize=10)#inline=True指字体在等高线中
    
    plt.xticks(())
    plt.yticks(())
    plt.show()
if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    # case5()
    # case6()
    case7()
    # case8()
    # case9()
    # case10()
