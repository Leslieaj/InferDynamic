import numpy as np
from scipy import linalg, linspace
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time

def fvdp2(t,y):
    y0, y1 = y   # y0是需要求解的函数，y1是一阶导
    # 注意返回的顺序是[一阶导， 二阶导]，这就形成了一阶微分方程组
    dydt = [y1, (1-y0**2)*y1-y0] 
    return dydt

t2 = linspace(0,20,1000)
tspan = (0.0, 20.0)
y0 = [[2,0]] # 初值条件
# 初值[2,0]表示y(0)=2,y'(0)=0
# 返回y，其中y[:,0]是y[0]的值，就是最终解，y[:,1]是y'(x)的值
y_ = solve_ivp(fvdp2, t_span=tspan, y0=y0[0], t_eval=t2)
print(y_.y.T)
print(np.random.normal(0,0.01))

