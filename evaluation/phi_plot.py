import matplotlib.pyplot as plt
import numpy as np


def phi_plot(phis, rewards, filename,outfile=False):
    avg_rewards = [np.mean(rwd) for rwd in rewards]
    std_rewards = [np.std(rwd) for rwd in rewards]
    plt.figure(figsize=(2*len(rewards),len(rewards)))
    plt.xlabel(r"$\phi$ value")
    plt.ylabel("average reward")
    plt.grid(True)
    plt.errorbar(phis, avg_rewards, yerr=std_rewards, fmt="d", capsize=5, label="SD")
    plt.plot(phis, avg_rewards, label="Mean lateness")
    # plt.axvline(x=np.pi,c="r",ls="--",label=r"$\pi$")
    # plt.axvline(x=2*np.pi,c="g",ls="--",label=r"$2\pi$")
    plt.legend()

    if not outfile:
        plt.show()
    else:
        plt.savefig("{}_PHI.pdf".format(filename))
