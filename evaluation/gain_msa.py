import numpy as np
import matplotlib.pyplot as plt

policies = 3
bar_width = 0.2
kpis = 4
# msa
lateness_msa = [2.8752,3.0023,1.0496]
wait_msa = [1.9429,2.0023,0.1600]
service_msa = [0.9324,0.9174,0.8896]
sys_load_msa = [0.2971,0.2929,0.2847]
# st
lateness = [2.7066,2.8398,0.8517]
wait = [1.8747,2.0033,0.0400]
service = [0.8319,0.8364,0.8117]
sys_load = [0.2656,0.2669,0.2597]
labs = ["5-Batch","5-Batch-One","1-Batch-1"]
kpis_colors = plt.cm.rainbow(np.linspace(0, 1, kpis))

fig = plt.figure()
all_axes = []
for i in range(policies):
    all_axes.append(fig.add_subplot(1, policies, i + 1))
props = dict(boxstyle='round', facecolor='darkorange')

for j, axes, labels in zip(range(policies), (all_axes), labs):
    gains = []
    for i, msa, st, color, lab in zip(range(kpis), (lateness_msa, wait_msa, service_msa, sys_load_msa),
                                      (lateness, wait, service, sys_load), kpis_colors,
                                      ("Lateness", "Wait Time", "Service Time", "Average System Load in %")):
        # gain = (msa[j] / st[j] - 1) * 100
        gain = (1- st[j]/msa[j]) * 100
        gains.append(gain)
        axes.bar(bar_width * i, gain, bar_width, label=lab, color=color)
    bars = axes.patches
    for bar, val in zip(bars, gains):
        height = bar.get_height() * 0.01
        axes.text(bar.get_x() + bar.get_width() / 2, height, "{}%".format(np.round(val, 2)), ha='center', va='bottom',
                  bbox=props)
    axes.legend()
    axes.set_xlabel("KPIs")
    axes.set_ylabel("Values")
    xticks = np.mean(axes.get_xlim())
    axes.set_xticks([xticks])
    axes.set_xticklabels([labels])
    axes.grid(True)

fig.set_size_inches(25, 5)
plt.savefig("opt_kpis_comp_gain.pdf")
