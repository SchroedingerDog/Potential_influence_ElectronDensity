import scipy.constants as const

mass = const.m_e  # 电子质量
ec = const.elementary_charge  # 元电荷
h = const.Planck  # 普朗克常数
eps0 = const.epsilon_0  # 真空电容率

def Hydrogen_energy(n_th):
    """n = 1, 2, 3, ... 完整的氢原子能级还需要角量子数E(n,l)"""
    return -mass * ec**4 / (8 * eps0**2 * h**2 * n_th**2)