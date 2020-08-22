import numpy as np
import time
from functools import wraps
from dynamics import ode_test
import experiment1, experiment2, experiment3, experiment4, experiment5
from infer_multi_ch import infer_model, test_model, simulation_ode_2, simulation_ode_3, test_model, test_model2, diff_method_backandfor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator

def case1():

    mode2 = experiment1.get_mode2(0)
    event1 = experiment1.get_event1(0)
    y0 = [[99.5,80],[97.5,100]]
    stepsize = 0.1
    maxorder = 1
    boundary_order = 1
    num_mode = 2
    T = 50
    ep = 0.01
    mergeep = 0.01
    method='piecelinear'
    t_list, y_list = simulation_ode_2(mode2, event1, y0, T, stepsize)
    with open('f1tr.txt','ab') as f:
        for ypoints in y_list:
            np.savetxt(f, ypoints, delimiter=" ") 
    P,G,C = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=mode2, event=event1, ep=ep, mergeep = mergeep, method=method, verbose=False)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    d_avg = test_model(
                P, G, C, num_mode, y_list , mode2, event1, maxorder, boundary_order)
    # d_avg = test_model2(
    #             P, G, C, num_mode, A, Y , mode2, event1, maxorder, boundary_order)
    print(G)
    print(C[0]/C[0],C[1]/C[0],C[2]/C[0])
    print(d_avg)
    @eventAttr()
    def eventtest(t,y):
        y0, y1 = y
        return C[0] * y0 + C[1] * y1 + C[2]
    
    ttest_list, ytest_list = simulation_ode_2([ode_test(G[0],maxorder),ode_test(G[1],maxorder)], eventtest, y0, T, stepsize)
    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='b',label='Original')
        else:
            plt.plot(y0_list,y1_list,c='b')
    for i, temp_y in enumerate(ytest_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='r', label='Inferred',linestyle='--')
        else:
            plt.plot(y0_list,y1_list,c='r',linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()



def case2():
    fvdp3 = experiment2.get_fvdp3(0)
    event1 = experiment2.get_event1(0)
    y0 = [[5,5,5], [2,2,2]]
    stepsize = 0.004
    maxorder = 2
    boundary_order = 1
    num_mode = 2
    T = 5
    ep = 0.01
    mergeep = 0.2
    method='tolmerge'
    t_list, y_list = simulation_ode_2(fvdp3, event1, y0, T, stepsize)
    # A, b1, b2, Y, ytuple = diff_method_backandfor(t_list, y_list, maxorder, stepsize)
    P,G,C = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=fvdp3, event=event1, ep=ep, mergeep= mergeep,method=method, verbose=False)
    d_avg = test_model(
                P, G, C, num_mode, y_list , fvdp3, event1, maxorder, boundary_order)
    # d_avg = test_model2(
    #             P, G, C, num_mode, A, Y , fvdp3, event1, maxorder, boundary_order)
    
    print(G)
    print(C[0]/C[0],C[1]/C[0],C[2]/C[0],C[3]/C[0])
    print(d_avg)
    with open('f2tr.txt','ab') as f:
        for ypoints in y_list:
            np.savetxt(f, ypoints, delimiter=" ") 
    
    @eventAttr()
    def eventtest(t,y):
        y0, y1, y2 = y
        return C[0] * y0 + C[1] * y1 + C[2]* y2 + C[3]
    
    ttest_list, ytest_list = simulation_ode_2([ode_test(G[0],maxorder),ode_test(G[1],maxorder)], eventtest, y0, T, stepsize)

    ax = plt.axes(projection='3d')
    for temp_y in y_list[0:1]:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        y2_list = temp_y.T[2]
        ax.plot3D(y0_list, y1_list, y2_list,c='b',label='Original')
    for temp_y in ytest_list[0:1]:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        y2_list = temp_y.T[2]
        ax.plot3D(y0_list, y1_list, y2_list,c='r',label='Inferred', linestyle='--')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.legend()
    plt.show()


def case5():
    modetr = experiment5.get_modetr(0)
    event = experiment5.get_event(0)
    labeltest = experiment5.get_labeltest(0)
    y0 = [[-1,1],[1,4],[2,-3]]
    T = 5
    stepsize = 0.01
    maxorder = 2
    boundary_order = 1
    num_mode = 3
    ep = 0.01
    mergeep=0.01
    method = 'piecelinear'
    t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)
    P, G, (coeff1, coeff2, [first,second,third]) = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=modetr, event=event, ep=ep, mergeep= mergeep,method=method, verbose=False,
                labeltest=labeltest)
    boundary = (coeff1, coeff2, [first,second,third])
    d_avg = test_model(
                P, G, boundary, num_mode, y_list, modetr, event, maxorder, boundary_order,
                labeltest=labeltest)
    print(d_avg)
    print(coeff1[0]/coeff1[0],coeff1[1]/coeff1[0],coeff1[2]/coeff1[0])
    print(coeff1[0]/coeff1[1],coeff1[1]/coeff1[1],coeff1[2]/coeff1[1])
    print(coeff2[0]/coeff2[0],coeff2[1]/coeff2[0],coeff2[2]/coeff2[0])
    print(coeff2[0]/coeff2[1],coeff2[1]/coeff2[1],coeff2[2]/coeff2[1])
    @eventAttr()
    def eventtest1(t,y):
        y0, y1 = y
        return coeff1[0] * y0 + coeff1[1] * y1 + coeff1[2]
    
    @eventAttr()
    def eventtest2(t,y):
        y0, y1 = y
        return coeff2[0] * y0 + coeff2[1] * y1 + coeff2[2]

    def labeltesttest(y):
        if eventtest1(0,y)>0:
            return first
        elif eventtest2(0,y)>0:
            return second
        else:
            return third

    ttest_list, ytest_list = simulation_ode_3([ode_test(G[0],maxorder),ode_test(G[1],maxorder),ode_test(G[2],maxorder)], [eventtest1,eventtest2], labeltesttest, y0, T, stepsize)
    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='b',label='Original')
        else:
            plt.plot(y0_list,y1_list,c='b')
    for i, temp_y in enumerate(ytest_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        if i == 0:
            plt.plot(y0_list,y1_list,c='r', label='Inferred',linestyle='--')
        else:
            plt.plot(y0_list,y1_list,c='r',linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # case1()
    # case2()
    case5()