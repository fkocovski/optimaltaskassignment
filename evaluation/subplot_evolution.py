import numpy as np
import matplotlib.pyplot as plt

filename = "WZ_ONE_TD_VFA_OP.csv"
delimiter = ","
skiprows = 1

data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
users = data.shape[1] - 9
unique_tasks = np.unique(data[:, 5])
task_colors = plt.cm.prism(np.linspace(0, 1, len(unique_tasks)))

a = np.zeros((4 * len(data[:, 5]), 3))

for i, t in enumerate(data):
    a[4 * i + 0, 0] = t[1]
    a[4 * i + 1, 0] = t[1]
    a[4 * i + 2, 0] = t[2]
    a[4 * i + 3, 0] = t[2]

    a[4 * i + 0, 1] = 0
    a[4 * i + 1, 1] = 1
    a[4 * i + 2, 1] = 0
    a[4 * i + 3, 1] = -1

    a[4 * i:4 * i + 4, 2] = t[5]


ax1 = plt.subplot(3, 1, 1)
ax1.set_ylabel("Global")

b = np.argsort(a[:, 0], kind="mergesort")
val = np.cumsum(a[b, 1])
ax1.fill_between(a[b, 0], val, facecolor="black")



# for i, t in enumerate(unique_tasks):
#     tmp = a[a[:, 2] == t]
#     b = np.argsort(tmp[:, 0], kind="mergesort")
#     val = np.cumsum(tmp[b, 1])
#     ax1.fill_between(tmp[b, 0], val, facecolor=task_colors[i])

for u in range(users):
    ax = plt.subplot(3, 1, u + 2, sharex=ax1)
    user_data = data[data[:, 4] == u + 1]
    u_array = np.zeros((4 * len(user_data[:, 5]), 3))
    for i, t in enumerate(user_data):
        u_array[4 * i + 0, 0] = t[2]
        u_array[4 * i + 1, 0] = t[2]
        u_array[4 * i + 2, 0] = t[3]
        u_array[4 * i + 3, 0] = t[3]

        u_array[4 * i + 0, 1] = 0
        u_array[4 * i + 1, 1] = 1
        u_array[4 * i + 2, 1] = 0
        u_array[4 * i + 3, 1] = -1

        u_array[4 * i:4*i+4, 2] = t[5]

    for i, t in enumerate(unique_tasks):
        tmp = u_array[u_array[:, 2] == t]
        b = np.argsort(tmp[:, 0], kind="mergesort")
        val = np.cumsum(tmp[b, 1])
        ax.fill_between(tmp[b, 0], val, facecolor=task_colors[i])
    ax.set_ylabel("User {}".format(u + 1))
    ylim = ax.get_ylim()
    ax.set_ylim(top=ylim[1]+1)


for t in data[:,1]:
    ax1.axvline(x=t,c="grey",ls="dotted",lw=0.5)

plt.show()
