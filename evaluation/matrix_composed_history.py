from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def matrix_composed_history(history):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(211, projection='3d')
    ax_two = fig.add_subplot(212, projection='3d')

    a_one = [x for x in history if x[1] == 0]
    a_two = [x for x in history if x[1] == 1]

    surf1 = ax.plot_trisurf([s1[0][0][0] for s1 in a_one], [s2[0][0][1] for s2 in a_one], [g[2] for g in a_one],
                            cmap=plt.cm.jet)
    surf2 = ax_two.plot_trisurf([s1[0][1][0] for s1 in a_two], [s2[0][1][1] for s2 in a_two], [g[2] for g in a_two],
                                cmap=plt.cm.jet)

    ax.set_xlabel('State 1')
    ax.set_ylabel('State 2')
    ax.set_zlabel('Reward')
    ax.set_title("Action 1")

    ax_two.set_xlabel('State 1')
    ax_two.set_ylabel('State 2')
    ax_two.set_zlabel('Reward')
    ax_two.set_title("Action 2")

    fig.colorbar(surf1, ax=ax)
    fig.colorbar(surf2, ax=ax_two)

    plt.show()
