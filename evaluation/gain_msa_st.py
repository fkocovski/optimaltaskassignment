import numpy as np
import matplotlib.pyplot as plt

policies = 3
bar_width = 0.2
kpis= 4
# msa
lateness_msa = [2.2460,2.3673,1.1191]
wait_msa = [1.5249,1.5602,0.1853]
service_msa = [0.7211,0.8072,0.9338]
sys_load_msa = [0.2394,0.2712,0.3362]
# st
lateness_st = [2.1942,2.1707,0.8093]
wait_st = [1.5157,1.4950,0.0295]
service_st = [0.6785,0.6756,0.7798]
sys_load_st = [0.2198,0.2054,0.2870]
kpis_colors = plt.cm.rainbow(np.linspace(0, 1, kpis))

fig = plt.figure()
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
props = dict(boxstyle='round', facecolor='darkorange')

for j,axes,labels in zip(range(policies),(ax,ax2,ax3),("5-Batch","5-Batch-1","1-Batch-1")):
    gains = []
    for i,msa,st,color,lab in zip(range(kpis),(lateness_msa,wait_msa,service_msa,sys_load_msa),(lateness_st,wait_st,service_st,sys_load_st),kpis_colors,("Lateness","Wait Time","Service Time","Average System Load in %")):
        gain = (msa[j]/st[j]-1)*100
        gains.append(gain)
        axes.bar(bar_width*i,gain,bar_width,label=lab,color=color)
    bars = axes.patches
    for bar,val in zip(bars,gains):
        height = bar.get_height()*0.75
        axes.text(bar.get_x() + bar.get_width() / 2, height, "{}%".format(np.round(val,2)), ha='center', va='bottom',bbox=props)
    axes.legend()
    axes.set_xlabel("KPIs")
    axes.set_ylabel("Values")
    xticks = np.mean(axes.get_xlim())
    axes.set_xticks([xticks])
    axes.set_xticklabels([labels])
    axes.grid(True)

fig.set_size_inches(25,5)
plt.savefig("opt_kpis_comp_gain.pdf")