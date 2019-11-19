import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def draw(t,y):
    """Draw
    """
    for temp_y in y:
        plt.plot(t,temp_y)
    plt.show()
    return 0

def draw2D(y):
    for temp_y in y:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        plt.plot(y0_list,y1_list)
    plt.show()
    return 0

def draw3D(y):
    ax = plt.axes(projection='3d')
    for temp_y in y:
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        y2_list = temp_y.T[2]
        ax.plot3D(y0_list, y1_list, y2_list)
    plt.show()
    return 0
    