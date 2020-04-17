import numpy as np
import time

import experiment1, experiment2, experiment3, experiment4, experiment5
from infer_multi_ch import infer_model

run = [1,2,3,4,5]


def run_test(eid, case_id, method, verbose=False):
    np.random.seed(0)

    if eid == 'A':
        case_info = experiment1.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        t_tuple = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment1.get_mode2(params)
        event = experiment1.get_event1(params)
        maxorder = 1
        boundary_order = 1
        num_mode = 2
        ep = 0.01
    
    elif eid == 'B':
        case_info = experiment2.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        t_tuple = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment2.get_fvdp3(params)
        event = experiment2.get_event1(params)
        maxorder = 2
        boundary_order = 1
        num_mode = 2
        ep = case_info['ep']

    start = time.time()
    a = infer_model(
        y0=y0, t_tuple=t_tuple, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
        num_mode=num_mode, modelist=modelist, event=event, ep=ep, method=method, verbose=verbose)
    end = time.time()
    print('eid:', eid, 'N_init:', len(t_tuple), 't_step:', stepsize, 'Method:', method)
    print('d_avg: %.6f, time: %.3f' % (a, end - start))
    return a


for i in range(4):
    for method in ['kmeans', 'merge', 'piecelinear']:
        run_test('A', i, method=method)

for i in range(4):
    for method in ['kmeans', 'merge', 'piecelinear']:
        run_test('B', i, method=method)


# if 1 in run:
#     # Isolette example
#     y0 = [[1,3],[-1,-2]]
#     t_tuple = [(0,20),(0,10)]

#     a = infer_model(
#         y0=y0, t_tuple=t_tuple, stepsize=0.01, maxorder=2, boundary_order=1,
#         modelist=experiment1.mode2, event=experiment1.event1, ep=0.01, method="new")
#     print(a)

# if 2 in run:
#     # Lorenz attractor
#     y0 = [[5,5,5], [2,2,2]]
#     t_tuple = [(0,15),(0,15)]

#     a = infer_model(
#         y0=y0, t_tuple=t_tuple, stepsize=0.01, maxorder=2, boundary_order=1,
#         modelist=experiment2.fvdp3, event=experiment2.event1, ep=0.01, method="new")
#     print(a)

# if 3 in run:
#     # Third degree ODE separated by parabola
#     y0 = [[0,1],[0,2],[0,3],[0,4],[0,5]]
#     t_tuple = [(0,20),(0,20),(0,20),(0,20),(0,20)]

#     a = infer_model(
#         y0=y0, t_tuple=t_tuple, stepsize=0.01, maxorder=3, boundary_order=2,
#         modelist=experiment3.mode, event=experiment3.event1, ep=0.01, method="new")
#     print(a)

# if 4 in run:
#     # Four dimensional case
#     y0 = [[4,0.1,3.1,0],[5.9,0.2,-3,0],[4.1,0.5,2,0],[6,0.7,2,0]]
#     t_tuple = [(0,5),(0,5),(0,5),(0,5)]

#     a = infer_model(
#         y0=y0, t_tuple=t_tuple, stepsize=0.01, maxorder=2, boundary_order=1,
#         modelist=experiment4.mmode, event=experiment4.event1, ep=0.01, method="new")
#     print(a)

# if 5 in run:
#     # Example with three modes
#     y0 = [[-1,1],[1,4],[2,-3]]
#     t_tuple = [(0,5),(0,5),(0,5)]
#     eventlist=[experiment5.eventtr_1,experiment5.eventtr_2,experiment5.eventtr_2]

#     a = infer_model(
#         y0=y0, t_tuple=t_tuple, stepsize=0.01, maxorder=2, boundary_order=1,
#         modelist=experiment5.modetr, event=eventlist, ep=0.01, method="new",
#         labeltest=experiment5.labeltest, num_mode=3)
#     print(a)
