import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=plt.figaspect(0.5))

filename = "1_BATCHONE_MSA_NU5_GI3_SIM1000"

path = "../simulations/optimization/batch/{}.csv".format(filename)
original_data = np.genfromtxt(path, delimiter=",", skip_header=1)

users = np.unique(original_data[:, 5])
sim_time = np.ceil(max(original_data[:, 4]))
user_loads = []
for i in range(len(users)):
    user_data = original_data[original_data[:, 5] == i + 1]
    user_loads.append(sum(user_data[:, 4] - user_data[:, 3]) / sim_time)
sys_load = np.mean(user_loads)

bars_colors = plt.cm.gray(np.linspace(0, 1, len(user_loads)))

ax2 = fig.add_subplot(111)
ax2.axhline(sys_load,c="darkorange",ls="dashed",label="Average system load: {:.2f} %".format(sys_load*100))
xt = [xtv for xtv in range(len(user_loads))]
xtl = ["User {}".format(user+1) for user in range(len(user_loads))]
ax2.bar(xt,user_loads,color=bars_colors,edgecolor="k",tick_label=xtl)
ax2.legend()
ax2.set_title("User Loads Over {} Time Units".format(int(sim_time)))


yt = np.arange(max(np.around(ax2.get_ylim(),decimals=1))+0.1,step=0.1)

ytl = ["{} %".format(int(val*100)) for val in yt]

ax2.set_yticks(yt)
ax2.set_yticklabels(ytl)

plt.savefig("{}_FAIR.png".format(filename))