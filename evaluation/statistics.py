import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_statistics(filename, outfile=False, delimiter=",", skip_header=1):
    original_data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)

    lateness = original_data[:, 4] - original_data[:, 1]
    wait = original_data[:, 3] - original_data[:, 1]
    service = original_data[:, 4] - original_data[:, 3]

    avg_lateness = np.average(lateness)
    avg_wait = np.average(wait)
    avg_service = np.average(service)
    n_of_jobs = original_data.shape[0]

    kpis = [lateness, wait, service]
    labels = ["Lateness", "Wait", "Service"]

    users = np.unique(original_data[:, 5])
    sim_time = np.ceil(max(original_data[:, 4]))
    user_loads = []
    for i in range(len(users)):
        user_data = original_data[original_data[:, 5] == i + 1]
        user_loads.append(sum(user_data[:, 4] - user_data[:, 3]) / sim_time)
    sys_load = np.mean(user_loads)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111)
    text = "Lateness: {:.4f}\nWait: {:.4f}\nService: {:.4f}\nJobs: {}\n$\phi$ load: {:.4f}".format(avg_lateness,
                                                                                                   avg_wait,
                                                                                                   avg_service,
                                                                                                   n_of_jobs, sys_load)

    props = dict(boxstyle='square', facecolor='darkorange')
    bplot = ax.boxplot(x=kpis, labels=labels, notch=True, patch_artist=True, sym="")
    ax.text(0.8,0.75 , text, transform=ax.transAxes, bbox=props)
    patches_colors = plt.cm.rainbow(np.linspace(0, 1, len(kpis)))

    for i, patch in enumerate(bplot["boxes"]):
        patch.set_facecolor(patches_colors[i])

    ax.grid(True)

    if not outfile:
        plt.show()
    else:
        file, ext = os.path.splitext(filename)
        plt.savefig("{}_KPI.pdf".format(file))
