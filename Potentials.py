import numpy as np 
import scipy.constants as const

ec = const.elementary_charge  # 元电荷
eps0 = const.epsilon_0  # 真空电容率
lenA = const.angstrom  # 埃A

def hydrogen_potential(x):
    """
    --------------------------------------------
    Note:
        类氢原子都是向心势 central potential
    """
    return -(ec**2) / (4 * np.pi * eps0 * np.abs(x))

class Barrier:
    @staticmethod
    def rectan_barr():
        pass

    @staticmethod
    def double_well(x, start0=-1, start1=1, width=1, magnitude=-1):
        """双势阱"""
        s0 = start0 * lenA
        s1 = start1 * lenA
        w = width * lenA
        m = magnitude * eV
        if width > 0:
            if start0 + width <= start1 or start1 + width <= start0:
                return m * (
                    np.heaviside(x - s0, 0.5)
                    - np.heaviside(x - s0 - w, 0.5)
                    + np.heaviside(x - s1, 0.5)
                    - np.heaviside(x - s1 - w, 0.5)
                )
            else:
                return "Not double_well !"
        else:
            return "Not double_well !"