import numpy as np
import matplotlib.pyplot as plt

policies = 5
bar_width = 0.2
kpis= 4
# msa
# lateness = [1.2160,1.1909,2.2460,2.3673,1.1191]
# wait = [0.0404,0.0009,1.5249,1.5602,0.1853]
# service = [1.1755,1.1900,0.7211,0.8072,0.9338]
# sys_load = [0.4279,0.4379,0.2394,0.2712,0.3362]
# st
lateness = [1.2160,1.1909,2.1942,2.1707,0.8093]
wait = [0.0404,0.0009,1.5157,1.4950,0.0295]
service = [1.1755,1.1900,0.6785,0.6756,0.7798]
sys_load = [0.4279,0.4379,0.2198,0.2054,0.2870]
index = np.arange(policies)
kpis_colors = plt.cm.rainbow(np.linspace(0, 1, kpis))

plt.figure(figsize=plt.figaspect(0.5))

for i,kpi,color,lab in zip(range(kpis),(lateness,wait,service,sys_load),kpis_colors,("Lateness","Wait Time","Service Time","Average System Load in %")):
    plt.bar(index+bar_width*i,kpi,bar_width,label=lab,facecolor=color)


plt.xlabel("KPIs")
plt.ylabel("Values")
plt.title("KPIs Comparison for Optimization Policies")
plt.xticks(index+bar_width+bar_width/2,("LLQP","SQ","5-Batch","5-Batch-One","1-Batch-1"))
plt.legend()
plt.grid(True)

plt.savefig("opt_kpis_comp.pdf")