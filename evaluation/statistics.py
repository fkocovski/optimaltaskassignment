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

    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax = fig.add_subplot(121)
    text = "Lateness: {:.4f}\nWait: {:.4f}\nService: {:.4f}\nJobs: {}".format(avg_lateness,avg_wait,avg_service,n_of_jobs)

    props = dict(boxstyle='round', facecolor='darkorange')
    ax.boxplot(x=kpis, labels=labels, notch=True, sym="")
    ax_ylims = ax.get_ylim()
    ax_xlims = ax.get_xlim()
    ax.text(0.75*max(ax_xlims),0.75*max(ax_ylims), text, bbox=props)


    ax.grid(True)

    bars_colors = plt.cm.gray(np.linspace(0, 1, len(user_loads)))


    ax2 = fig.add_subplot(122)
    # ax2.axhline(sys_load,c="darkorange",ls="dashed",lw=0.5,label="Average system load: {:.2f} %".format(sys_load*100))
    ax2.axhline(sys_load,c="darkorange",ls="dashed",label="Average system load: {:.2f} %".format(sys_load*100))
    xt = [xtv for xtv in range(len(user_loads))]
    xtl = ["User {}".format(user+1) for user in range(len(user_loads))]
    ax2.bar(xt,user_loads,color=bars_colors,edgecolor="k",tick_label=xtl)
    ax2.legend()


    yt = np.arange(max(np.around(ax2.get_ylim(),decimals=1))+0.1,step=0.1)

    ytl = ["{} %".format(int(val*100)) for val in yt]

    ax2.set_yticks(yt)
    ax2.set_yticklabels(ytl)

    if not outfile:
        plt.show()
    else:
        file, ext = os.path.splitext(filename)
        plt.savefig("{}_KPI.pdf".format(file))
