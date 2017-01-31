import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def qsa_values(value_action, outfile=None):
    """
Plots the corresponding QSA values for two users based on the trajectory's history.
    :param value_action: list containing tuples in the form (states, qsa_value, action).
    :param outfile: if provided, it outputs the plot to a file.
    """
    x = []
    y = []
    z = []

    for states, value, action in value_action:
        x.append(states[0])
        y.append(states[1])
        z.append(value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x, y, z, cmap=plt.cm.jet)

    ax.set_xlabel('User 1 Busy Time')
    ax.set_ylabel('User 2 Busy Time')
    ax.set_zlabel('QSA Value')
    fig.colorbar(surf)

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
