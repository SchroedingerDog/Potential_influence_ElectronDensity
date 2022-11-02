"""
基于GPU的向量化运算，将波函数表示为离散格点的向量，将微分算子化为矩阵。
将定态薛定谔方程的求解化为哈密顿矩阵的本征值、本征向量求解问题。
"""

import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import constants as const

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> The hardware running is {dev}.")

mass = const.m_e  # 电子质量
lenA = const.angstrom  # 埃A
hbar = const.hbar  # 约化普朗克常数


class Schrodinger_1d:
    """
    --------------------------------------------
    Description:
        对角化一维定态薛定谔方程 Diagonalize 1d stationary Schrodinger equation

    --------------------------------------------
    属性 Attributes:
        self.X      ==>  继承的是输入的位置范围
        self.V      ==>  将势能函数离散化，并写成矩阵对角元
        self.L      ==>  Laplace算符的离散化
        self.H      ==>  Hamiltonian的离散化
        self.eig_E  ==>  系统的本征值
        self.eig_S  ==>  系统的本征态

    --------------------------------------------
    方法 Methods:
        Laplacian       ==>  将Laplace算符离散化为矩阵
        eigen_sol       ==>  返回从大到小排列的本征值和本征态
        eigen_value     ==>  输入参数n，返回第n个本征值
        eigen_energy    ==>  输入参数n，返回第n个本征态
        plot_potential  ==>  画出势能项
        plot_wavefunc   ==>  输入参数n，画出第n个波函数
        plot_probable   ==>  输入参数n，画出第n个概率函数
        plot_levels     ==>  输入参数n，画出前n个能级
        plot_check      ==>  输入参数n，画出第n个线性变换结果对比
    """

    def __init__(self, potential, x_min=-50, x_max=50, sites=1_000):
        self.X = torch.linspace(x_min * lenA, x_max * lenA, steps=sites)  # X在cpu上
        self.V = torch.diag(potential(self.X)).to(dev)  # 注意势能函数要能够传入tensor
        self.L = self.Laplacian(self.X[1] - self.X[0], sites).to(dev)
        self.H = -(hbar**2) / (2 * mass) * self.L + self.V
        self.eig_E, self.eig_S = self.eigen_sol(self.H)  # torch.linalg.eigh(self.H)

    @staticmethod
    def Laplacian(x_step, x_num):
        """
        --------------------------------------------
        Args:
            x_step: float -> the duration or interval of x axis
            x_num:  int   -> the number of x axis discrete points

        --------------------------------------------
        Return:
            二阶导数的有限差分矩阵表示
        """
        D_2st = (
            torch.diag(tensor([1] * (x_num - 1)), diagonal=1)
            + torch.diag(tensor([1] * (x_num - 1)), diagonal=-1)
            + torch.diag(tensor([-2] * x_num))
        )
        return D_2st.to(dev) / (x_step**2)

    @staticmethod
    def eigen_sol(square_matrix):
        """
        --------------------------------------------
        Description:
            特征向量是矩阵v的列，即满足 M @ v[:,i] = w[i] * v[:,i]

        --------------------------------------------
        Args:
            square_matrix -> must be Hermitian or symmetric matrices

        --------------------------------------------
        Note:
            实对称矩阵特征值一定是实数，可以比较大小
        """
        w, v = torch.linalg.eigh(square_matrix)  # 特征值是升序排列的
        return w, v

    def eigen_state(self, n_th):
        """
        --------------------------------------------
        Args:
            n_th < sites

        --------------------------------------------
        Return:
            第n个本征态矢量
        """
        return self.eig_S[:, n_th]

    def eigen_energy(self, n_th):
        """
        --------------------------------------------
        Args:
            n_th < sites

        --------------------------------------------
        Return:
            第n个本征能量
        """
        return self.eig_E[n_th]

    @property
    def plot_potential(self):
        """
        --------------------------------------------
        Note:
            要返回图形show()
        """
        plt.plot(self.X, self.V.diag().cpu(), label="Potential")
        plt.ylabel(r"V $(Joule)$")
        plt.xlabel(r"$x$")
        plt.legend(loc="best")
        plt.show()

    def plot_wavefunc(self, n_th=0):
        """
        --------------------------------------------
        Note:
            不返回图形，只作为图元素
            把多个波函数画一张图里再show()
        """
        leg = r"$E_{%s}=%.2e$" % (n_th, self.eig_E[n_th].item())
        plt.plot(self.X, self.eigen_state(n_th).cpu(), label=leg)
        plt.xlabel("$x$")
        plt.ylabel(r"$\psi_{%s}(x)$" % n_th)
        plt.legend(loc="best")

    def plot_probable(self, n_th=0):
        """画概率密度函数 rho = psi^H @ psi"""
        rho = np.abs(self.eigen_state(n_th)) ** 2
        # 格式化输出%.2e保留2位小数位，使用科学计数法
        leg = r"$E_{%s}=%.2e$" % (n_th, self.eig_E[n_th])
        plt.plot(self.X, rho, label=leg)
        plt.ylabel(r"$\rho_{%s}(x)=\psi_{%s}^*(x)\psi_{%s}(x)$" % (n_th, n_th, n_th))
        plt.xlabel(r"$x$")
        plt.legend(loc="best")

    def plot_levels(self, num_levels=10):
        for i in range(num_levels):
            plt.plot(self.X, np.ones_like(self.X) * self.eig_E[i], "r-", linewidth=0.5)
        plt.ylabel(r"Energy levels (J)")
        plt.xlabel(r"$x$")

    def plot_compare(self, n_th=0):
        """本征态检验 H | psi_n == E_n | psi_n 画图对比"""
        Ham_psi = self.H @ self.eigen_state(n_th)
        Egy_psi = self.eigen_value(n_th) * self.eigen_state(n_th)
        plt.plot(self.X, Ham_psi, label=r"$H|\psi_{%s} \rangle$" % n_th)
        plt.plot(self.X, Egy_psi, "-.", label=r"$E |\psi_{%s} \rangle$" % n_th)
        plt.legend(loc="upper right")
        plt.xlabel(r"$x$")
        plt.ylim(Egy_psi.min(), Egy_psi.max() * 1.6)
