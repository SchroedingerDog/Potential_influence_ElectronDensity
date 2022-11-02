import scipy.constants as const
import torch
from torch import tensor
from scipy.special import hermite  # 埃尔米特多项式
from math import factorial  # 阶乘

mass = const.m_e  # 电子质量
ec = const.elementary_charge  # 元电荷
h = const.Planck  # 普朗克常数
hbar = const.hbar  # 约化普朗克常数
eps0 = const.epsilon_0  # 真空电容率


def Hydrogen_energy(n_th):
    """n = 1, 2, 3, ... 完整的氢原子能级还需要角量子数E(n,l)"""
    return -mass * ec**4 / (8 * eps0**2 * h**2 * n_th**2)


class Oscillator:
    """量子谐振子的波函数和能级理论值"""

    lambd = 5897 * lenA
    omega = 2 * torch.pi * vc / lambd

    def __init__(self, x_min=-50, x_max=50, sites=1_000):
        self.X = torch.linspace(x_min * lenA, x_max * lenA, sites)
        self.dx = self.X[1] - self.X[0]

    @classmethod
    def analytic_energy(cls, n_th=0):
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
