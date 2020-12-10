import numpy as np
import time
from functools import wraps
from dynamics import ode_test
import experiment1, experiment2, experiment3, experiment4, experiment5
from infer_multi_ch import infer_model, test_model, simulation_ode_2, simulation_ode_3, test_model, test_model2, diff_method_backandfor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import math
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
        y0_list = temp_y.T[0][::5]
        y1_list = temp_y.T[1][::5]
        if i == 0:
            plt.plot(y0_list,y1_list,mfc='None', mec='r',label='Inferred',marker='.',linestyle='None')
            
        else:
            # plt.plot(y0_list,y1_list,c='r',linestyle='--',marker=',')
            plt.plot(y0_list,y1_list,mfc='None', mec='r',marker='.',linestyle='None')
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


def casef():
    x_data = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13', '14', '15','16', '17','18','19','20']
    y_data = [0.003992015, 0.003193576, 0.003984067, 0.000000002, 33266.837080135, 0.343082352, 4.102138097,0.566123868,0.001746934,0.019164384,0.000398672,0.080461907,0.009204978,0.065775879,0.014511614,0.051080809,43.683964703,270.121113172,0.213860637,0.192239074]
    # y_data = [0.003992015, 0.003193576, 0.003984067, 0.000000002, 0, 0.343082352, 4.102138097,0.566123868,0.001746934,0.019164384,0.000398672,0.080461907,0.009204978,0.065775879,0.014511614,0.051080809,0,0,0.213860637,0.192239074]
    y_data2 = [0.001996009, 0.000000001, 0.000000004,0.000000002, 4.590208463, 0.343082301, 4.102138524,0.566648476,0.000683030,0.019164384,0.000398672,0.080462003,0.009204710,0.065775879,0.014511629,0.051100292,0.069413185,0.044627401,0.213969149,0.192343534]
    y_data3 = [0.001996009, 0.000000001, 0.003983424,0.000000002, 3.441166109, 0.139407793, 0.182332361,0.568632158,0.000001165,0.017742012,0.000434219,0.062894223,0.021369155,0.008614517,0.014511627,0.043977445,0.069407697,0.128860837,0.213922570,0.192325895]
    bar_width=0.3
    # plt.bar(x=range(len(x_data)), height=y_data, label='method 1',
    # color='steelblue', alpha=0.8, width=bar_width)
    Y_data = [math.log(10,x) for x in y_data]
    Y_data2 = [math.log(10,x) for x in y_data2]
    Y_data3 = [math.log(10,x) for x in y_data3]

    plt.bar(x=np.arange(len(x_data)), height=Y_data,
    label='method 1', color='b', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(len(x_data))+bar_width, height=Y_data2,
    label='method 2', color='r', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(len(x_data))+bar_width + bar_width, height=Y_data3,
    label='method 3', color='g', alpha=0.8, width=bar_width)

    # for x, y in enumerate(y_data):
    #     plt.text(x, y + 100, '%s' % y, ha='center', va='top')
    # for x, y in enumerate(Y_data2):
    #     plt.text(x, y, '%s' % y, ha='center', va='top')
    # for x, y in enumerate(Y_data3):
    #         plt.text(x+bar_width, y, '%s' % y, ha='center', va='top')
    plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)
    plt.xlabel('label')
    plt.ylabel('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    case1()
    # case2()
    # case5()
    # casef()