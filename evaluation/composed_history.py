from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def composed_history(history):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(121, projection='3d')
    ax_two = fig.add_subplot(122, projection='3d')

    a_one = [(x[0], x[2]) for x in history if x[1] == 0]
    a_two = [(x[0], x[2]) for x in history if x[1] == 1]

    surf1 = ax.plot_trisurf([s1[0][0] for s1 in a_one], [s2[0][1] for s2 in a_one], [g[1] for g in a_one],
                            cmap=plt.cm.jet)
    surf2 = ax_two.plot_trisurf([s1[0][0] for s1 in a_two], [s2[0][1] for s2 in a_two], [g[1] for g in a_two],
                                cmap=plt.cm.jet)

    ax.set_xlabel('busy time 1')
    ax.set_ylabel('busy time 2')
    ax.set_zlabel('G value')
    ax.set_title("Action 1")

    ax_two.set_xlabel('busy time 1')
    ax_two.set_ylabel('busy time 2')
    ax_two.set_zlabel('G value')
    ax_two.set_title("Action 2")

    fig.colorbar(surf1, ax=ax)
    fig.colorbar(surf2, ax=ax_two)

    plt.show()
