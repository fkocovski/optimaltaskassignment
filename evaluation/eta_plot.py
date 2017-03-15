import matplotlib.pyplot as plt
import numpy as np


def eta_plot(etas, rewards, outfile=None):
    avg_rewards = [np.mean(rwd) for rwd in rewards]
    std_rewards = [np.std(rwd) for rwd in rewards]
    plt.xlabel("eta value")
    plt.ylabel("average reward")
    plt.grid(True)
    plt.errorbar(etas, avg_rewards, yerr=std_rewards, fmt="d", capsize=5, label="SD")
    plt.plot(etas, avg_rewards, label="Mean lateness")
    plt.legend()

    # TODO change outfile to be boolean as other plot scripts
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
