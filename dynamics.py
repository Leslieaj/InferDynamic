#The dynamic functions.
import numpy as np
from fractions import Fraction

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

def mode2_1_test(result_coef):
    def ode(t,y):
        y0, y1 = y
        ode1_coef = result_coef[0]
        ode2_coef = result_coef[1]
        dydt = [ode1_coef.dot(np.array([y0**2, y0*y1, y1**2, y0,y1,1])), ode2_coef.dot(np.array([y0**2, y0*y1, y1**2, y0,y1,1]))]
        return dydt
    return ode
