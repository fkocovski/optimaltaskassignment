import numpy as np
import matplotlib.pyplot as plt

policy = "BATCHONE"
solver = "ST"
nu = 5
gi = 3
sim = 1000
folder = "batch"
outfile = True

avg_lateness = []
avg_wait = []
avg_service = []

for i in range(10):
    try:
        df = np.genfromtxt("../simulations/optimization/{}/{}_{}_{}_NU{}_GI{}_SIM{}.csv".format(folder,i + 1, policy, solver,nu,gi,sim), delimiter=",",
                        skip_header=1)
        avg_lateness.append(np.average(df[:,4]-df[:,1]))
        avg_wait.append(np.average(df[:,3]-df[:,1]))
        avg_service.append(np.average(df[:,4] - df[:,3]))
    except OSError:
        print("Files only up to batch size {} for solver {}".format(i, solver))
        break


max_length = max(len(avg_lateness), len(avg_wait), len(avg_service))

x = [i + 1 for i in range(max_length)]
x_labels = ["{}_{}".format(i + 1, policy) for i in range(max_length)]
plt.figure(figsize=plt.figaspect(0.25))
plt.plot(x, avg_lateness, label="Average lateness", ls="--")
plt.plot(x, avg_service, label="Average service time", ls="-.")
plt.plot(x, avg_wait, label="Average waiting time", ls=":")
plt.grid(True)
plt.legend()
plt.xticks(np.arange(1, max_length + 1), x_labels)


if not outfile:
    plt.show()
else:
    plt.savefig("../simulations/optimization/{}/1-{}_{}_{}_NU{}_GI{}_SIM{}_BSEVAL.pdf".format(folder,max_length,policy,solver,nu,gi,sim))
