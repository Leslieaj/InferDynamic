import numpy as np
import time

import experiment1, experiment2, experiment3, experiment4, experiment5
from infer_multi_ch import infer_model, test_model, simulation_ode_2, simulation_ode_3


total_win, total_d_avg, total_time = dict(), dict(), dict()
methods = ['kmeans', 'merge', 'piecelinear']
for method in methods:
    total_win[method] = 0
    total_d_avg[method] = 0.0
    total_time[method] = 0.0


def run_test(id, eid, case_id, methods, verbose=False):
    np.random.seed(0)

    if eid == 'A':
        case_info = experiment1.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        y0_test = case_info['y0_test']
        T = case_info['t_tuple']
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
        y0_test = case_info['y0_test']
        T = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment2.get_fvdp3(params)
        event = experiment2.get_event1(params)
        maxorder = 2
        boundary_order = 1
        num_mode = 2
        ep = case_info['ep']

    elif eid == 'C':
        case_info = experiment3.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        y0_test = case_info['y0_test']
        T = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment3.get_mode(params)
        event = experiment3.get_event(params)
        maxorder = 3
        boundary_order = 2
        num_mode = 2
        ep = case_info['ep']

    elif eid == 'D':
        case_info = experiment4.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        y0_test = case_info['y0_test']
        T = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment4.get_mmode(params)
        event = experiment4.get_event(params)
        maxorder = 2
        boundary_order = 1
        num_mode = 2
        ep = case_info['ep']

    elif eid == 'E':
        case_info = experiment5.cases[case_id]
        params = case_info['params']
        y0 = case_info['y0']
        y0_test = case_info['y0_test']
        T = case_info['t_tuple']
        stepsize = case_info['stepsize']
        modelist = experiment5.get_modetr(params)
        event = experiment5.get_event(params)
        labeltest = experiment5.get_labeltest(params)
        maxorder = 2
        boundary_order = 1
        num_mode = 3
        ep = case_info['ep']

    # Obtain simulated trajectory
    start = time.time()
    if num_mode == 2:
        t_list, y_list = simulation_ode_2(modelist, event, y0, T, stepsize)
        test_t_list, test_y_list = simulation_ode_2(modelist, event, y0_test, T, stepsize)
    elif num_mode == 3:
        t_list, y_list = simulation_ode_3(modelist, event, labeltest, y0, T, stepsize)
        test_t_list, test_y_list = simulation_ode_3(modelist, event, labeltest, y0_test, T, stepsize)
    else:
        raise NotImplementedError
    end = time.time()

    # print('eid:', eid, 'N_init:', len(y0), 't_step:', stepsize, 'ep:', ep, 'sim_time: %.3f' % (end - start))

    d_avg = dict()
    infer_time = dict()
    for method in methods:
        start = time.time()
        if num_mode == 2:
            P, G, boundary = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=modelist, event=event, ep=ep, method=method, verbose=verbose)
            end = time.time()
            d_avg[method] = test_model(
                P, G, boundary, num_mode, y_list + test_y_list, modelist, event, maxorder, boundary_order)
            infer_time[method] = end - start
        elif num_mode == 3:
            P, G, boundary = infer_model(
                t_list, y_list, stepsize=stepsize, maxorder=maxorder, boundary_order=boundary_order,
                num_mode=num_mode, modelist=modelist, event=event, ep=ep, method=method, verbose=verbose,
                labeltest=labeltest)
            end = time.time()
            d_avg[method] = test_model(
                P, G, boundary, num_mode, y_list + test_y_list, modelist, event, maxorder, boundary_order,
                labeltest=labeltest)
            infer_time[method] = end - start
        else:
            raise NotImplementedError

        # print('Method: %s, d_avg: %.6f, infer_time: %.3f' % (method, d_avg[method], infer_time[method]))

    best_method, best_avg = None, 1.0
    for method, avg in d_avg.items():
        total_d_avg[method] += avg
        if avg < best_avg:
            best_method, best_avg = method, avg
    for method, t in infer_time.items():
        total_time[method] += t
    total_win[best_method] += 1

    print('%d & $%s$ & %d & %.2f & %d & - & %.5f & %.5f & - & %.1f & %.1f \\\\' % (
        id, eid, len(y0), stepsize, T, d_avg['merge'], d_avg['piecelinear'],
        infer_time['merge'], infer_time['piecelinear']))
    return d_avg, infer_time


for i in range(4):
    run_test(i+1, 'A', i, methods=['kmeans','merge', 'piecelinear'])

# for i in range(4):
#     run_test(i+5, 'B', i, methods=['merge', 'piecelinear'])

# for i in range(4):
#     run_test(i+9, 'C', i, methods=['merge', 'piecelinear'])

# for i in range(4):
#     run_test(i+13, 'D', i, methods=['merge', 'piecelinear'])

# for i in range(4):
#     run_test(i+17, 'E', i, methods=['merge', 'piecelinear'])

print('total win:', total_win)
print('total d_avg:', total_d_avg)
print('total time:', total_time)

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
