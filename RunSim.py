from Solver import Schrodinger_1d
from Potentials import Hydrogen, Barrier, Oscillator
import matplotlib.pyplot as plt


def test_1():
    example = Schrodinger_1d(Hydrogen, x_min=-1, x_max=1)
    example.plot_potential


def test_2():
    p = lambda x: Barrier.double_well(x, start0=-10, start1=7, magnitude=-0.45, width=3)
    example = Schrodinger_1d(p, x_min=-100, x_max=100, sites=3000)
    example.plot_potential


def test_3():
    p = lambda x: Barrier.double_well(x, start0=-10, start1=7, magnitude=-0.45, width=3)
    example = Schrodinger_1d(p, x_min=-100, x_max=100, sites=3000)
    plt.figure(dpi=150, figsize=(12, 5), num="Double well probs")
    example.plot_probable(0)
    example.plot_probable(1)
    example.plot_probable(2)
    example.plot_probable(3)
    plt.show()


def test_4():
    example = Schrodinger_1d(Oscillator)
    example.plot_wavefunc(1)
    example.plot_wavefunc(2)
    example.plot_wavefunc(3)
    plt.show()


def test_5():
    p = lambda x: Barrier.double_well(x, start0=-10, start1=7, magnitude=-0.45, width=3)
    example = Schrodinger_1d(p, x_min=-100, x_max=100, sites=3000)
    example.plot_levels(num_levels=15)
    plt.show()


if __name__ == "__main__":
    # test_1()
    # test_2()
    # test_3()
    test_5()
