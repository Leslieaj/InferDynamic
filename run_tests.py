import experiment1, experiment2, experiment3, experiment4, experiment5

run = [1,2,3,4,5]

if 1 in run:
    # Isolette example
    y0 = [[1,3],[-1,-2]]
    t_tuple = [(0,20),(0,10)]
    stepsize = 0.01
    maxorder = 2

    a = experiment1.case(y0,t_tuple,stepsize,maxorder,experiment1.mode2,experiment1.event1,0.01,"new")
    print(a)

if 2 in run:
    # Lorenz attractor
    y0 = [[5,5,5], [2,2,2]]
    t_tuple = [(0,15),(0,15)]
    stepsize = 0.01
    maxorder = 2

    a = experiment2.case(y0,t_tuple,stepsize,maxorder,experiment2.fvdp3,experiment2.event1,0.01,"new")
    print(a)

if 3 in run:
    # Third degree ODE separated by parabola
    y0 = [[0,1],[0,2],[0,3],[0,4],[0,5]]
    t_tuple = [(0,20),(0,20),(0,20),(0,20),(0,20)]
    stepsize = 0.01
    maxorder = 3

    a = experiment3.case(y0,t_tuple,stepsize,maxorder,experiment3.mode,experiment3.event1,0.01,"new")
    print(a)

if 4 in run:
    # Four dimensional case
    y0 = [[4,0.1,3.1,0],[5.9,0.2,-3,0],[4.1,0.5,2,0],[6,0.7,2,0]]
    t_tuple = [(0,5),(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2

    a = experiment4.case(y0,t_tuple,stepsize,maxorder,experiment4.mmode,experiment4.event1,0.01,"new")
    print(a)

if 5 in run:
    # Example with three modes
    y0 = [[-1,1],[1,4],[2,-3]]
    t_tuple = [(0,5),(0,5),(0,5)]
    stepsize = 0.01
    maxorder = 2
    eventlist=[experiment5.eventtr_1,experiment5.eventtr_2,experiment5.eventtr_2]

    a = experiment5.case(y0,t_tuple,stepsize,maxorder,experiment5.modetr,eventlist,experiment5.labeltest,0.01,"new")
    print(a)
