import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def qsa_values(value_action,outfile=None):

    x = []
    y = []
    z = []

    for states,value,action in value_action:
        x.append(states[action])
        y.append(np.mean(states))
        z.append(value)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x,y,z,cmap=plt.cm.jet)

    ax.set_xlabel('Chosen User Busy Time')
    ax.set_ylabel('Average Busy Time')
    ax.set_zlabel('QSA Value')
    fig.colorbar(surf)

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()