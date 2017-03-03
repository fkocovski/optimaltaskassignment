import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_statistics(file, outfile=None):
    """
Calculates statistics for passed file.
    :param outfile: a string file containing the name of the file where the plot file is saved. If not provided, the plot is shown.
    :param file: string of the file name.
    """
    file = open(file)

    df = pd.read_csv(file, sep=",")

    file.close()

    df["lateness"] = df["finished"] - df["arrival"]
    df["wait"] = df["started"] - df["arrival"]
    df["service"] = df["finished"] - df["started"]

    avg_lateness = np.average(df["lateness"])
    avg_wait = np.average(df["wait"])
    avg_service = np.average(df["service"])
    n_of_jobs = len(df["job"])

    users = len(df["user"].unique())
    sim_time = np.ceil(df["finished"].iloc[-1])
    user_loads = []
    for i in range(users):
        user_loads.append(sum(df["service"].where(df["user"] == i+1).dropna())/sim_time)
    sys_load = np.mean(user_loads)

    df = df.sort_values("arrival")

    df.to_csv("{}".format(file.name), index=False)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111)
    text = "Lateness: {:.4f}\nWait: {:.4f}\nService: {:.4f}\nJobs: {}\n$\phi$ load: {:.4f}".format(avg_lateness, avg_wait, avg_service,
                                                                              n_of_jobs,sys_load)
    props = dict(boxstyle='square', facecolor='darkorange', alpha=0.75)
    ax.boxplot([df["lateness"], df["wait"], df["service"]], labels=["Lateness", "Wait", "Service"], showmeans=True,
               meanline=True, sym="")
    ax.text(0.8, 0.75, text, transform=ax.transAxes, bbox=props)
    ax.set_title(file.name)
    ax.grid(True)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
