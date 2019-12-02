#The dynamic functions.
import numpy as np
from fractions import Fraction
from generator import generate_complete_polynomail

def dydx1(t,y):
    dy_dx = -1.34*y**3+2.7*y**2-4*y+5.6
    return dy_dx

def dydx2(t,y):
    dy_dx = -1.34*y**3+9.8*y**2+6.5*y-23
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
        dydt = [-0.26*(y0-y1), -1.0]
    else:
        dydt = [-0.26*(y0-y1), 1.0]
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
    if y0 > 50:
        dydt = [-y0**2 + 51*y0 + 3,1]
    elif y0 > 20:
        dydt = [y0 + 3,1]
    else:
        dydt = [-y0 + 43,1]
    return dydt

def conti_test1(t,y):
    y0, y1 = y
    if y0 > 20:
        dydt = [15*y0 - 100,1]
    else:
        dydt = [-y0**2 + 25*y0 + 100,1]
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