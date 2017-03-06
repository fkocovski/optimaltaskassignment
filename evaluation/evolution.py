import numpy as np
import matplotlib.pyplot as plt
import itertools

def _filter_arrivals(D):
    n = D.shape[0]
    c = np.sum(D[:,1:], axis=1)
    for i in range(1,n):
        if c[i] > c[i-1]:
            yield D[i,0]


def evolution(filename, outfile=None, delimiter=",", skiprows=1, title=None):
    """
    Plot the simulated system state over time.
    """

    # load data
    D = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
    #D = np.genfromtxt(filename, dtype=float, delimiter=delimiter, names=True)

    # basic figure setup
    nusers = D.shape[1]-3
    H = [max(int(np.max(D[:,i+1])),1) for i in range(nusers+1)]  # heights of 'subplots' (global queue/user queues)
    P = np.hstack((0, np.cumsum([-H[i+1]-1 for i in range(nusers)])))  # 'subplot offsets'
    plt.figure(figsize=((np.max(D[:,0])-np.min(D[0,:]))/2.0, (H[0]-P[-1]+1.0)/2.0))

    unique_tasks = np.unique(D[:,-1])
    task_colors = plt.cm.prism(np.linspace(0, 1, len(unique_tasks)))

    # plot pending tasks
#    F = np.hstack((True,np.diff(D[:,0],2)!=0.0,True)) # filter 'zero duration' points; assuming random event times
    plt.fill_between(D[:,0], D[:,1], 0.0, linewidth=0.5, facecolor="grey",edgecolor="grey")
    plt.axhline(y=0.0, c="k", lw=0.8)

    # plot users
    # for i,t in enumerate(unique_tasks):
    #     for j in range(nusers):
    #         plt.fill_between(D[:, 0], D[:, j + 2] + P[j + 1], P[j + 1],where=D[:,-1]==t, linewidth=0.5, facecolor=task_colors[i], edgecolor=task_colors[i],interpolate=True)
    #         plt.axhline(y=P[j+1], c="k", lw=0.8)

    # plot users
    for i in range(nusers):
        plt.fill_between(D[:, 0], D[:, i + 2] + P[i + 1], P[i + 1], linewidth=0.5, facecolor="green",
                         edgecolor="black")
        plt.axhline(y=P[i + 1], c="k", lw=0.8)

    # plot arrival events
    for t in _filter_arrivals(D):
        plt.axvline(x=t, c="k", ls="dotted", lw=0.5)

    # axis ticks and labels
    plt.gca().set_xlabel("time")
    plt.gca().set_xlim(0.0, np.ceil(D[-1,0]))
    plt.gca().set_ylim(P[-1]-0.5, H[0]+0.5)
    plt.gca().set_yticks([P[i]+hi for i in range(nusers+1) for hi in range(H[i],-1,-1)])
    plt.gca().set_yticklabels(["%d" % hi for i in range(nusers+1) for hi in range(H[i],-1,-1)])
    plt.gca().set_yticks([P[i]+H[i]/2.0 for i in range(nusers+1)], minor=True)
    plt.gca().set_yticklabels(["pending      "] + ["user %d      " % (i+1) for i in range(nusers)], minor=True)
    #plt.gca().set_yticklines([], minor=True)
    for line in plt.gca().yaxis.get_minorticklines():
        line.set_visible(False)
    #plt.grid(axis="y", lw=0.5, ls="--")
    # plt.ylabel('y')
    if title is not None:
        plt.title(title)
    # plt.legend()

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()