import numpy as np
import time
from functools import wraps
from dynamics import ode_test
import experiment1, experiment2, experiment3, experiment4, experiment5
from infer_multi_ch import infer_model, test_model, simulation_ode_2, simulation_ode_3
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
    t_list, y_list = simulation_ode_2(mode2, event1, y0, T, stepsize)
    P,G,C = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=mode2, event=event1, ep=ep, method='merge', verbose=False)
    

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
    stepsize = 0.002
    maxorder = 2
    boundary_order = 1
    num_mode = 2
    T = 5
    ep = 0.01
    t_list, y_list = simulation_ode_2(fvdp3, event1, y0, T, stepsize)
    P,G,C = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=fvdp3, event=event1, ep=ep, method='merge', verbose=False)
    
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
        ax.plot3D(y0_list, y1_list, y2_list,c='b')
    for temp_y in ytest_list[0:1]:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        y2_list = temp_y.T[2]
        ax.plot3D(y0_list, y1_list, y2_list,c='r',linestyle='--')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
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
    t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)
    P, G, (coeff1, coeff2, [first,second,third]) = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=modetr, event=event, ep=ep, method='piecelinear', verbose=False,
                labeltest=labeltest)

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