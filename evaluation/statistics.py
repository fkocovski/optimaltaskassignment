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
    df["avg_lateness"] = avg_lateness
    df["avg_wait"] = avg_wait
    df["avg_service"] = avg_service
    df["n_of_jobs"] = n_of_jobs

    df = df.sort_values("arrival")

    df.to_csv("eval_{}".format(file.name), index=False)

    df.plot(y=["lateness", "wait", "service"], kind="kde",grid=True)


    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
