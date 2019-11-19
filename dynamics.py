#The dynamic functions.
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