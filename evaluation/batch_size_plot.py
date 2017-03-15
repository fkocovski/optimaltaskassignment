import numpy as np
import matplotlib.pyplot as plt

sys_load = 0.5
n_users = 5
policy = "KBatch"
solver = "ST"
outfile = "{}_avg_{}_{}_{}.pdf".format(policy, solver, sys_load, n_users)
batch_type = "batch"

avg_lateness = []
avg_wait = []
avg_service = []

for i in range(10):
    try:
        df = np.loadtxt("../simulations/eval_{}{}_{}.csv".format(i + 1, batch_type, solver), delimiter=",",
                        skiprows=1)
        avg_lateness.append(df[0][-4])
        avg_wait.append(df[0][-3])
        avg_service.append(df[0][-2])
    except FileNotFoundError:
        print("Files only up to batch size {} for solver {}".format(i, solver))
        break

max_length = max(len(avg_lateness), len(avg_wait), len(avg_service))

x = [i + 1 for i in range(max_length)]
x_labels = ["{}{}".format(i + 1, batch_type) for i in range(max_length)]
plt.figure(figsize=(1.5 * max_length, 0.5 * max_length))
plt.plot(x, avg_lateness, label="Average lateness", ls="--")
plt.plot(x, avg_service, label="Average service time", ls="-.")
plt.plot(x, avg_wait, label="Average waiting time", ls=":")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(1, max_length + 1), x_labels)
plt.title("Sys. load: {}, Users: {}, Policy: {}, Solver: {}".format(sys_load, n_users, policy, solver))


# TODO change outfile to be boolean as other plot scripts
if outfile is None:
    plt.show()
else:
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
