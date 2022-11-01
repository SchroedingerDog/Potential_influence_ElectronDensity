class Schrodinger_1d:
    """
    --------------------------------------------
    Description:
    对角化一维定态薛定谔方程 Diagonalize 1d stationary Schrodinger equation

    属性 Attributes:
        self.X      ==>  继承的是输入的位置范围
        self.V      ==>  将势能函数离散化，并写成矩阵对角元
        self.L      ==>  Laplace算符的离散化
        self.H      ==>  Hamiltonian的离散化
        self.eig_E  ==>  系统的本征值
        self.eig_S  ==>  系统的本征态

    方法 Methods:
        Laplacian       ==>  将Laplace算符离散化为矩阵
        eigen_sol       ==>  返回从大到小排列的本征值和本征态
        eigen_value     ==>  输入参数n，返回第n个本征值
        eigen_state     ==>  输入参数n，返回第n个本征态
        plot_potential  ==>  画出势能项
        plot_wavefunc   ==>  输入参数n，画出第n个波函数
        plot_probable   ==>  输入参数n，画出第n个概率函数
        plot_levels     ==>  输入参数n，画出前n个能级
        plot_check      ==>  输入参数n，画出第n个线性变换结果对比
    """

    def __init__(self, potential, x_min=-50, x_max=50, sites=1_000):
        self.X = np.linspace(start=x_min * lenA, stop=x_max * lenA, num=sites)
        self.V = np.diag(potential(self.X))
        self.L = self.Laplacian(self.X[1] - self.X[0], sites)
        self.H = -(hbar**2) / (2 * mass) * self.L + self.V
        self.eig_E, self.eig_S = self.eigen_sol(self.H)

    @staticmethod
    def Laplacian(x_step, x_num):
        """x_step is the duration or interval of x axis"""
        D_2st = (
            np.diag([1] * (x_num - 1), k=1)
            + np.diag([1] * (x_num - 1), k=-1)
            + np.diag([-2] * x_num)
        )
        return D_2st / (x_step**2)  # 2阶微分的差分格式

    @staticmethod
    def eigen_sol(square_matrix):
        """特征向量是矩阵v的列
        M @ v[:,i] = w[i] * v[:,i]
        """
        val, vec = np.linalg.eig(square_matrix)
        idx_sort = np.argsort(val)  # 从小到大排序
        return val[idx_sort], vec[:, idx_sort]

    def eigen_state(self, n_th):
        """输出第n个本征态矢量 n < sites"""
        return self.eig_S[:, n_th]

    def eigen_value(self, n_th):
        """输出第n个本征能量 n < sites"""
        return self.eig_E[n_th]

    @property
    def plot_potential(self):
        plt.plot(self.X, np.diag(self.V))
        plt.ylabel(r"V $(J)$")
        plt.xlabel(r"$x$")
        plt.show()

    def plot_wavefunc(self, n_th=0):
        leg = r"$E_{%s}=%.2e$" % (n_th, self.eig_E[n_th])
        plt.plot(self.X, self.eigen_state(n_th), label=leg)
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