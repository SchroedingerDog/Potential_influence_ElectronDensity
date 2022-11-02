import numpy as np
import torch
from torch import tensor
import scipy.constants as const

vc = const.c  # 真空光速
lenA = const.angstrom  # 埃A
mass = const.m_e  # 电子质量
ec = const.elementary_charge  # 元电荷
eps0 = const.epsilon_0  # 真空电容率
eV = const.physical_constants["electron volt"][0]  # 电子伏特


def Hydrogen(x):
    """
    --------------------------------------------
    Note:
        类氢原子都是向心势 central potential
    """
    return -(ec**2) / (4 * torch.pi * eps0 * torch.abs(x))


class Barrier:
    @staticmethod
    def rectan_barr():
        """
        --------------------------------------------
        Description:
            方势垒
        """
        pass

    @staticmethod
    def double_well(x, start0=-1, start1=1, width=1, magnitude=-1):
        """
        --------------------------------------------
        Description:
            双势阱
        """
        s0 = start0 * lenA
        s1 = start1 * lenA
        w = width * lenA
        m = magnitude * eV
        if width > 0:
            if start0 + width <= start1 or start1 + width <= start0:
                return m * (
                    torch.heaviside(x - s0, tensor([0.5]))
                    - torch.heaviside(x - s0 - w, tensor([0.5]))
                    + torch.heaviside(x - s1, tensor([0.5]))
                    - torch.heaviside(x - s1 - w, tensor([0.5]))
                )
            else:
                return "Not double_well !"
        else:
            return "Not double_well !"


def Oscillator(x):
    """
    --------------------------------------------
    Description:
        quantum harmonic oscillating
    """
    lambd = 5897 * lenA
    omega = 2 * torch.pi * vc / lambd
    return mass * omega**2 * x**2 / 2


class oscillator:
    """
    --------------------------------------------
    Description:
        quantum harmonic oscillating
    """

    lambd = 5897 * lenA
    omega = 2 * torch.pi * vc / lambd

    def __init__(self, x_min=-50, x_max=50, sites=1_000):
        self.X = torch.linspace(x_min * lenA, x_max * lenA, sites)
        self.dx = self.X[1] - self.X[0]

    @classmethod
    def potential(cls, x):
        return mass * cls.omega**2 * x**2 / 2

    @classmethod
    def analytic_eg(cls, n_th=0):
        """谐振子能级的解析解"""
        return hbar * cls.omega * (n_th + 1 / 2)

    @classmethod
    def analytic_wf(cls, x, n_th=0):
        """谐振子波函数的解析解"""
        coef = 1 / np.sqrt(2**n_th * factorial(n_th))
        coef = coef * (mass * cls.omega / hbar / np.pi) ** 0.25
        Hn = hermite(n_th)
        wf = (
            coef
            * np.exp(-mass * cls.omega * x**2 / (2 * hbar))
            * Hn(np.sqrt(mass * cls.omega / hbar) * x)
        )
        return wf

    def plot_probable(self, n_th=0):
        psi = self.analytic_wf(self.X, n_th)
        rho = np.abs(psi) ** 2 * self.dx
        plt.plot(self.X, rho, "r:", label="analytic")
        plt.legend(loc="best")
