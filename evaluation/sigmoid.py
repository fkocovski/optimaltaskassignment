import matplotlib.pyplot as plt


def sigmoid(t, sig, tp, sigp,name, outfile=False):
    plt.figure(figsize=plt.figaspect(0.25))
    plt.subplot(121)
    plt.plot(t, sig,label="Generation interval",c="darkorange")
    plt.axvline(max(t)/2,c="k",ls="dashed",lw=0.5,label="Sigmoid's midpoint")
    plt.xlabel("Training time")
    plt.grid(True)
    plt.legend()

    plt.subplot(122)
    plt.plot(tp, sigp,label="Epsilon",c="darkorange")
    plt.axvline(max(tp)/2,c="k",ls="dashed",lw=0.5,label="Sigmoid's midpoint")
    plt.xlabel("Training time")
    plt.grid(True)
    plt.legend()

    if not outfile:
        plt.show()
    else:
        plt.savefig("{}_SIG.pdf".format(name))
