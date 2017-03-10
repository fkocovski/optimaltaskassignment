import numpy as np
import matplotlib.pyplot as plt
import os


def fill_array(data, start_index, finish_index, task_id):
    array = np.zeros((4 * len(data[:, 5]), 3))
    for i, t in enumerate(data):
        array[4 * i + 0, 0] = t[start_index]
        array[4 * i + 1, 0] = t[start_index]
        array[4 * i + 2, 0] = t[finish_index]
        array[4 * i + 3, 0] = t[finish_index]
        if t[6] == task_id:
            array[4 * i + 0, 1] = 0
            array[4 * i + 1, 1] = 1
            array[4 * i + 2, 1] = 0
            array[4 * i + 3, 1] = -1

    sorted_index = np.argsort(array[:, 0], kind="mergesort")
    values = np.cumsum(array[sorted_index, 1])

    return sorted_index, values, array


def evolution(filename, outfile=False, delimiter=",", skip_header=1):
    original_data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)

    task_names = np.genfromtxt(filename,delimiter=delimiter,skip_header=skip_header,usecols=(6,7),dtype=None)
    unique_names = dict(np.unique(task_names))

    users = len(np.unique(original_data[:, 5]))
    unique_tasks = np.unique(original_data[:, 6])
    task_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tasks)))
    ax1 = plt.subplot(users + 1, 1, 1)
    ax1.set_ylabel("Global")
    old_global_values = 0.0
    for i, task_id in enumerate(unique_tasks):
        global_sorted_index, global_values, global_array = fill_array(original_data, 1, 2, task_id)
        ax1.fill_between(x=global_array[global_sorted_index, 0], y1=old_global_values + global_values,
                         y2=old_global_values,
                         facecolor=task_colors[i], label=unique_names[task_id].decode("UTF-8"))
        old_global_values += global_values
    plt.xticks(np.arange(int(min(original_data[:, 1])), int(max(original_data[:, 4])), 10.0))
    ylims = ax1.get_ylim()
    ax1.set_yticks(np.arange(0, int(ylims[1]) + 1, 1))
    for a, ass in original_data[:, [1, 2]]:
        ax1.axvline(x=a, c="k", ls="dotted", lw=0.5)
        ax1.axvline(x=ass, c="k", ls="dashdot", lw=0.5)
    handles, labels = ax1.get_legend_handles_labels()
    assigned_artist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='dotted',linewidth=0.5)
    finished_artist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='dashdot',linewidth=0.5)
    ax1.legend(handles + [assigned_artist, finished_artist], labels + ["Arrival", "Assigned"])

    for u in range(users):
        ax = plt.subplot(users + 1, 1, u + 2, sharex=ax1)
        user_data = original_data[original_data[:, 5] == u + 1]
        old_user_values = 0.0
        for i, task_id in enumerate(unique_tasks):
            sorted_user_index, user_values, user_array = fill_array(user_data, 2, 4, task_id)
            ax.fill_between(x=user_array[sorted_user_index, 0], y1=old_user_values + user_values, y2=old_user_values,
                            facecolor=task_colors[i], label=unique_names[task_id].decode("UTF-8"))
            old_user_values += user_values
        handles, labels = ax.get_legend_handles_labels()
        assigned_artist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='dotted',linewidth=0.5)
        finished_artist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='dashdot',linewidth=0.5)
        ax.legend(handles+[assigned_artist,finished_artist],labels+["Assigned","Finished"])
        ylims = ax.get_ylim()
        ax.set_yticks(np.arange(0, int(ylims[1]) + 1, 1))
        ax.set_ylabel("User {}".format(u + 1))
        for a, f in user_data[:, [2, 4]]:
            ax.axvline(x=a, c="k", ls="dotted", lw=0.5)
            ax.axvline(x=f, c="k", ls="dashdot", lw=0.5)



    all_axes = plt.gcf().get_axes()

    max_y = max(y.get_ylim()[1] for y in all_axes)
    max_x = max(x.get_xlim()[1] for x in all_axes)

    plt.gcf().set_size_inches(max_x / 2.0, (len(all_axes) * max_y))

    if not outfile:
        plt.show()
    else:
        file,ext = os.path.splitext(filename)
        plt.savefig("{}_EVO.pdf".format(file))
