import matplotlib.pyplot as plt
import numpy as np


def rewards_plot(rewards, outfile=None):
    plt.xlabel("Time")
    plt.ylabel("Rewards")
    plt.grid(True)
    plt.plot(range(len(rewards)), rewards, label="Rewards over time")
    plt.legend()

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
