
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    # plot pending tasks
#    F = np.hstack((True,np.diff(D[:,0],2)!=0.0,True)) # filter 'zero duration' points; assuming random event times
    plt.fill_between(D[:,0], D[:,1], 0.0, linewidth=0.5, facecolor="darkorange",edgecolor="black")
    plt.axhline(y=0.0, c="k", lw=0.8)

    # plot users
    for i in range(nusers):
        plt.fill_between(D[:,0], D[:,i+2]+P[i+1], P[i+1], linewidth=0.5, facecolor="green",edgecolor="black")
        plt.axhline(y=P[i+1], c="k", lw=0.8)

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


def pending(filename, outfile=None, delimiter=",", skiprows=1, title=None):
    """
    Plot share of simulation time over pending tasks.
    """

    # load data
    D = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
    #D = np.genfromtxt(filename, dtype=float, delimiter=delimiter, names=True)

    # count pending tasks
    nusers = D.shape[1]-2
    x = np.sum(np.hstack((D[:-1,[1]], np.maximum(D[:-1,2:]-1.0, 0.0))), axis=1, dtype=int)
    w = np.diff(D[:,0])/D[-1,0]
    c = np.bincount(x, w)

    # basic figure setup
    plt.figure(figsize=(len(c)+2, 5))

    # plot
    plt.bar(np.array(range(len(c))), c, width=0.8, edgecolor="#204a87", facecolor="#729fcf")
    pmean = np.dot(range(len(c)),c)
    lmean = plt.axvline(x=pmean, c="#a40000")

    # axis ticks and labels
    plt.gca().set_xlabel("pending tasks")
    plt.gca().set_xlim(-1.0, len(c))
    plt.gca().set_xticks(range(len(c)))
    plt.gca().set_ylabel("time share")
    plt.gca().set_ylim(0.0, 1.0)
    plt.legend([lmean], ["mean: %.3f" % pmean])
    if title is not None:
        plt.title(title)

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")


def service(filename, outfile=None, delimiter=",", title=None):
    """
    Plot histogram of task service times.
    """

    # load data
    df = pd.read_csv(filename, sep=delimiter, header=0)

    # compute service time of tasks
    tf = pd.merge(
            df.loc[df["event"]==0,("node","token","time")].rename(columns={"time": "arrival"}),
            df.loc[df["event"]==2,("node","token","user","time")].rename(columns={"time": "finish"}),
            how="inner", on=("node","token"))
    tf = pd.concat((tf, tf["finish"].sub(tf["arrival"])), axis=1).rename(columns={0: "time"})

    # compute histogram
    t = tf["time"].values
    tmax = np.ceil(10.0*np.max(t))/10.0
    nbins = int(10.0*tmax)
    c, _ = np.histogram(t, np.linspace(0.0, tmax, num=nbins+1))
    c = c.astype(float)  # /np.sum(c)

    # basic figure setup
    plt.figure(figsize=(len(c)+2, 5))

    # plot
    plt.bar(np.linspace(0.0, tmax*(nbins-1.0)/nbins, num=nbins), c, width=tmax/nbins, align="edge", edgecolor="#204a87", facecolor="#729fcf")
    tmean = np.mean(t)
    lmean = plt.axvline(x=tmean, c="#a40000")

    # axis ticks and labels
    plt.gca().set_xlabel("service time")
    #plt.gca().set_xlim(-0.5, tmax+0.5)
    #plt.gca().set_xticks(range(tmax+1))
    plt.gca().set_ylabel("number of tasks")
    #plt.gca().set_ylim(0.0, 1.0)
    plt.legend([lmean], ["mean: %.3f" % tmean])
    if title is not None:
        plt.title(title)

    # output
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
