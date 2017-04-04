import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(121)
ax_two = fig.add_subplot(122)

x = np.arange(0,6)
x_labels = ["leaving\noffice","reach\ncar","exiting\nhighway","2ndary\nroad","home\nstreet","arrive\nhome"]
y = [30,40,35,40,45,45]
update_mc = [(45,0,1),(45,1,2),(45,2,3),(45,3,4)]
arrow_mc = [(0,30,0,45),(1,40,0,45),(2,35,0,45),(3,40,0,45)]
update_td = [(40,0,1),(35,1,2),(40,2,3),(45,3,4)]
arrow_td = [(0,30,0,40),(1,40,0,35),(2,35,0,40),(3,40,0,45)]

for axes,title,hline,arrow in zip([ax,ax_two],["Monte Carlo","Temporal Difference"],[update_mc,update_td],[arrow_mc,arrow_td]):
    axes.plot(x,y,color="black")
    axes.scatter(x,y,marker="D",color="darkorange",zorder=100)
    for val,i1,i2 in hline:
        axes.hlines(val, i1,i2, linestyles="dashed")
    for x0,y0,x1,y1 in arrow:
        axes.arrow(x0,y0,x1,y1-y0,head_width=0.3,length_includes_head=True,fc="darkorange")
    axes.set_xticks(x)
    axes.set_xticklabels(x_labels)
    axes.set_ylabel("Predited total travel time")
    axes.set_title(title)

plt.savefig("mc_td.pdf")