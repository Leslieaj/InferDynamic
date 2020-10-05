import numpy as np
from fractions import Fraction
from generator import generate_complete_polynomial
from functools import wraps
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
from sklearn.svm import SVR
from scipy.linalg import pinv, pinv2
from scipy.integrate import odeint, solve_ivp
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
from infer_multi_ch import simulation_ode, diff_method_new, diff_method_new_6
from dynamics import fvdp3_1

def case():
    y0 = [[5,5,5], [2,2,2]]
    stepsize = 0.01
    maxorder = 2
    T = 5
    t_list, y_list = simulation_ode(fvdp3_1, y0, T, stepsize, eps=0.02)
    A,b,Y = diff_method_new(t_list,y_list,maxorder,stepsize)
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(A, b)
    g = clf.coef_
    print(g)
    A,b,Y = diff_method_new_6(t_list,y_list,maxorder,stepsize)
    clf.fit(A, b)
    g = clf.coef_
    print(g)

if __name__ == "__main__":
    case()