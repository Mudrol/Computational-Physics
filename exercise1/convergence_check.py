import numpy as np
from num_calculus import first_derivative, simpson_int
import matplotlib.pyplot as plt


def my_plot(fig, x, f, xlabel="", ylabel="", title="", linestyle=""):
    """
    This function plots a single mathematical function with custom
    xlabel, ylabel and title.
    """

    plt.rcParams["legend.handlelength"] = 2
    plt.rcParams["legend.numpoints"] = 1
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    ax = fig.add_subplot(111)

    ax.plot(x, f, linestyle, label=title)

    # include legend (with best location, i.e., loc=0)
    ax.legend(loc=0)

    # set axes labels and limits
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x.min(), x.max())
    fig.tight_layout(pad=1)


def main():
    """
    Calculating and plotting the absolute error for the first derivative
    of sin(x) at x = 2 and for the Simpson integral of sin(x) from 0 to 1.
    """
    x = 2
    def f(x): return np.sin(x)
    dx = np.linspace(0.0001, 0.1, 1000)

    # First derivative absolute error
    abs_error = np.zeros(len(dx))

    # Calculate all the derivatives for grid spacings
    for i in range(len(dx)):
        df = first_derivative(f, x, dx[i])
        abs_error[i] = abs(df-np.cos(2))

    # Calculate all of the integrals with same spacing as first derivative
    abs_error_simpson = np.zeros(len(dx))
    for i in range(len(dx)):

        # Amount of points is the inverse of spacing
        x = np.linspace(0, 1, int(1./dx[i]))
        fs = np.sin(x)
        Is = simpson_int(fs, x)
        abs_error_simpson[i] = abs(Is-(1-np.cos(1)))

    # Create a figure and plot both of the results into same figure
    fig1 = plt.figure(1)
    my_plot(fig1, dx, abs_error, "", "", "First derivative", "--")
    my_plot(fig1, dx, abs_error_simpson, "$dx$", "$Error$", "Simpson integral")

    plt.show()


if __name__ == "__main__":
    main()
