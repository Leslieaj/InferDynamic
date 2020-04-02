#The dynamic functions.
import numpy as np
from fractions import Fraction
from generator import generate_complete_polynomail

model = 0

def dydx1(t,y):
    dy_dx = -1.34*y**3+2.7*y**2-4*y+5.6
    return dy_dx

def dydx2(t,y):
    dy_dx = -1.34*y**3+9.8*y**2+6.5*y-23
    return dy_dx

def dydx3(t,y):
    dy_dx = y-10
    return dy_dx

def fvdp2(t,y):
    y0, y1 = y   # y0是需要求解的函数，y1是一阶导
    # 注意返回的顺序是[一阶导， 二阶导]，这就形成了一阶微分方程组
    dydt = [y1, (1-y0**2)*y1-y0] 
    return dydt

def fvdp2_1(t,y):
    y0, y1 = y
    dydt = [2*y0-y0*y1, 2*y0**2-y1]
    return dydt

def fvdp3_1(t,y):
    y0, y1, y2 = y
    sigma = 10
    beta = Fraction(8,3)
    rho = 28
    dydt  = [sigma*(y1-y0), y0*(rho-y2)-y1, y0*y1-beta*y2]
    return dydt

def mode2_1(t,y):
    """ A hybrid automaton with 2 modes for an incubator.
    """
    y0, y1 = y
    if y0 > 0:
        dydt = [-0.26*(y0-y1), -0.5]
    else:
        dydt = [-0.26*(y0-y1), 0.5]
    return dydt


def mode2_11(t,y):
    """ A hybrid automaton with 2 modes for an incubator.
    """
    global model
    y0, y1 = y

    if y0 > 1:
        model = 1
    elif y0 < 0:
        model = 0


    if model == 0:
        dydt = [1.0, 1.0]
    elif model == 1:
        dydt = [-1.0, 1.0]
    return dydt


def mode2_1_test(result_coef,order):
    def ode(t,y):
        y0, y1 = y
        ode1_coef = result_coef[0]
        ode2_coef = result_coef[1]
        A = generate_complete_polynomail(2,order)
        basicf = []
        for i in range(0,A.shape[0]):
            basicf.append(y0**A[i][0]*y1**A[i][1])
        b = np.array(basicf)
        dydt = [ode1_coef.dot(b), ode2_coef.dot(b)]
        return dydt
    return ode

def conti_test(t,y):
    y0, y1 = y
    if y0 > 5:
        dydt = [-y0**2 + 5*y0 + 11,1]
    elif y0 > 2:
        dydt = [y0 + 2.5 ,1]
    else:
        dydt = [-y0 + 3,1]
    return dydt

def conti_test1(t,y):
    y0, y1 = y
    if y0 > 2:
        dydt = [-y0 +4,1]
    else:
        dydt = [y0,1]
    return dydt

def conti_test_test(result_coef,order):
    def ode(t,y):
        y0, y1 = y
        ode1_coef = result_coef[0]
        ode2_coef = result_coef[1]
        A = generate_complete_polynomail(2,order)
        basicf = []
        for i in range(0,A.shape[0]):
            basicf.append(y0**A[i][0]*y1**A[i][1])
        b = np.array(basicf)
        dydt = [ode1_coef.dot(b), ode2_coef.dot(b)]
        return dydt
    return ode


def ode_test(result_coef,order):
    def ode(t,y):
        # print("in")
        dim = result_coef.shape[0]
        A = generate_complete_polynomail(dim,order)
        basicf = []
        for i in range(0,A.shape[0]):
            ap = 1
            for j in range(0,A.shape[1]):
                ap = ap*(y[j]**A[i][j])
            basicf.append(ap)
        b = np.array(basicf)
        dydt = np.zeros(dim)
        for l in range(0,dim):
            dydt[l] = result_coef[l].dot(b)
        # print("out")
        return dydt
    return ode

         
    
    