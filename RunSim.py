from Solver import Schrodinger_1d
from Potentials import Hydrogen, Barrier, Oscillator
import matplotlib.pyplot as plt


def test_1():
    example = Schrodinger_1d(Hydrogen, x_min=-1, x_max=1)
    example.plot_potential


def test_2():
    example = Schrodinger_1d(Barrier.double_well, x_min=-10, x_max=10)
    example.plot_potential


def test_3():
    example = Schrodinger_1d(Barrier.double_well, x_min=-10, x_max=10, sites=4000)
    example.plot_wavefunc(1)
    example.plot_wavefunc(2)
    example.plot_wavefunc(3)
    plt.show()


def test_4():
    example = Schrodinger_1d(Oscillator)
    example.plot_wavefunc(1)
    example.plot_wavefunc(2)
    example.plot_wavefunc(3)
    plt.show()


if __name__ == "__main__":
    # test_1()
    # test_2()
    test_4()
