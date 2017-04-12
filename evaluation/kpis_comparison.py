import numpy as np
import matplotlib.pyplot as plt

policies = 5
bar_width = 0.2
kpis= 4
# msa
# lateness = [1.2809,1.2135,2.8752,3.0023,1.0496]
# wait = [0.0732,0.0037,1.9429,2.0023,0.1600]
# service = [1.2077,1.2098,0.9324,0.9174,0.8896]
# sys_load = [0.3860,0.3866,0.2971,0.2929,0.2847]
# st
lateness = [1.2809,1.2135,2.7066,2.8398,0.8517]
wait = [0.0732,0.0037,1.8747,2.0033,0.0400]
service = [1.2077,1.2098,0.8319,0.8364,0.8117]
sys_load = [0.3860,0.3866,0.2656,0.2669,0.2597]
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

plt.savefig("opt_st_kpis_comp.pdf")