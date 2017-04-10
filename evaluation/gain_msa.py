import numpy as np
import matplotlib.pyplot as plt

policies = 1
bar_width = 0.2
kpis = 4
# 1batch1
lateness_msa = [1.3138]
wait_msa = [0.0874]
service_msa = [1.2264]
sys_load_msa = [0.3924]
# other
lateness = [1.3422]
wait = [0.1408]
service = [1.2014]
sys_load = [0.3840]
labs = ["llqp_mc_vfa"]
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
        gain = (msa[j] / st[j] - 1) * 100
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