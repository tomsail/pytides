import operator as op
import string
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from . import nodal_corrections as nc
from .astro import AstronomicalParameter


class BaseConstituent(object):
    """Base class for constituent

    Parameters
    ----------
    object : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    xdo_int: Dict[str, int] = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": -8,
        "S": -7,
        "T": -6,
        "U": -5,
        "V": -4,
        "W": -3,
        "X": -2,
        "Y": -1,
        "Z": 0,
    }

    int_xdo: Dict[int, str] = {v: k for k, v in xdo_int.items()}

    def __init__(
        self,
        name: str,
        xdo: str = "",
        coefficients: List[int] = [],
        u: Callable[[Any], float] = nc.u_zero,
        f: Callable[[Any], float] = nc.f_unity,
    ) -> None:
        if xdo == "":
            self.coefficients: np.ndarray = np.array(coefficients)
        else:
            self.coefficients = np.array(self.xdo_to_coefficients(xdo))
        self.name = name
        self.u = u
        self.f = f

    def xdo_to_coefficients(self, xdo: str) -> List[int]:
        return [self.xdo_int[l.upper()] for l in xdo if l in string.ascii_letters]

    def coefficients_to_xdo(self, coefficients: List[int]) -> str:
        return "".join([self.int_xdo[c] for c in coefficients])

    def V(self, astro: np.ndarray) -> np.ndarray:
        return np.dot(self.coefficients, self.astro_values(astro))

    def xdo(self) -> str:
        return self.coefficients_to_xdo(self.coefficients)

    def speed(self, a: Dict[str, AstronomicalParameter]) -> np.ndarray:
        return np.dot(self.coefficients, self.astro_speeds(a))

    def astro_xdo(
        _, a: Dict[str, AstronomicalParameter]
    ) -> List[AstronomicalParameter]:
        return [a["T+h-s"], a["s"], a["h"], a["p"], a["N"], a["pp"], a["90"]]

    def astro_speeds(self, a: Dict[str, AstronomicalParameter]) -> np.ndarray:
        return np.array([each.speed for each in self.astro_xdo(a)])

    def astro_values(self, a: Dict[str, AstronomicalParameter]) -> np.ndarray:
        return np.array([each.value for each in self.astro_xdo(a)])

    # Consider two out of phase constituents which travel at the same speed to
    # be identical
    def __eq__(self, c: Any) -> Any:
        return np.all(self.coefficients[:-1] == c.coefficients[:-1])

    def __hash__(self) -> int:
        return hash(tuple(self.coefficients[:-1]))


class CompoundConstituent(BaseConstituent):
    def __init__(
        self, members: Optional[List[Tuple[BaseConstituent, int]]] = None, **kwargs: Any
    ) -> None:
        if members is None:
            raise ValueError

        self.members = members

        if "u" not in kwargs:
            kwargs["u"] = self.u
        if "f" not in kwargs:
            kwargs["f"] = self.f

        super(CompoundConstituent, self).__init__(**kwargs)

        self.coefficients = reduce(op.add, [c.coefficients * n for (c, n) in members])

    def speed(self, a: Dict[str, AstronomicalParameter]) -> Union[float, int]:
        return sum([n * c.speed(a) for (c, n) in self.members])

    def V(self, a: Dict[str, AstronomicalParameter]) -> Union[float, int]:
        return sum([n * c.V(a) for (c, n) in self.members])

    def u(self, a: Dict[str, AstronomicalParameter]) -> Union[float, int]:
        return sum([n * c.u(a) for (c, n) in self.members])

    def f(self, a: Dict[str, AstronomicalParameter]) -> Union[Any, float, int]:
        return np.prod([c.f(a) ** abs(n) for (c, n) in self.members])


###### Base Constituents
# Long Term
_Z0 = BaseConstituent(name="Z0", xdo="Z ZZZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_Sa = BaseConstituent(name="Sa", xdo="Z ZAZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_Ssa = BaseConstituent(name="Ssa", xdo="Z ZBZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_MSm = BaseConstituent(name="MSm", xdo="Z AXA ZZZ", u=nc.u_zero, f=nc.f_Mm)
_Mm = BaseConstituent(name="Mm", xdo="Z AZY ZZZ", u=nc.u_zero, f=nc.f_Mm)
_Mf = BaseConstituent(name="Mf", xdo="Z BZZ ZZZ", u=nc.u_Mf, f=nc.f_Mf)

# Diurnals
_Q1 = BaseConstituent(name="Q1", xdo="A XZA ZZA", u=nc.u_O1, f=nc.f_O1)
_O1 = BaseConstituent(name="O1", xdo="A YZZ ZZA", u=nc.u_O1, f=nc.f_O1)
_K1 = BaseConstituent(name="K1", xdo="A AZZ ZZY", u=nc.u_K1, f=nc.f_K1)
_J1 = BaseConstituent(name="J1", xdo="A BZY ZZY", u=nc.u_J1, f=nc.f_J1)

# M1 is a tricky business for reasons of convention, rather than theory.  The
# reasons for this are best summarized by Schureman paragraphs 126, 127 and in
# the comments found in congen_input.txt of x tides, so I won't go over all this
# again here.

_M1 = BaseConstituent(name="M1", xdo="A ZZZ ZZA", u=nc.u_M1, f=nc.f_M1)
_P1 = BaseConstituent(name="P1", xdo="A AXZ ZZA", u=nc.u_zero, f=nc.f_unity)
_S1 = BaseConstituent(name="S1", xdo="A AYZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_OO1 = BaseConstituent(name="OO1", xdo="A CZZ ZZY", u=nc.u_OO1, f=nc.f_OO1)

# Additional minor diurnals. The lunar members are grouped with the major
# constituent sharing their node-factor structure (O1 for the lower-frequency
# group, J1 for the upper group, OO1 for the highest); the solar members
# (pi1, psi1) require no nodal correction.
_alpha1 = BaseConstituent(name="alpha1", xdo="A VBA ZZA", u=nc.u_O1, f=nc.f_O1)
_sigma1 = BaseConstituent(name="sigma1", xdo="A WBZ ZZA", u=nc.u_O1, f=nc.f_O1)
_tau1 = BaseConstituent(name="tau1", xdo="A YBZ ZZA", u=nc.u_O1, f=nc.f_O1)
_beta1 = BaseConstituent(name="beta1", xdo="A ZXA ZZA", u=nc.u_O1, f=nc.f_O1)
_NO1 = BaseConstituent(name="NO1", xdo="A ZZA ZZA", u=nc.u_J1, f=nc.f_J1)
_chi1 = BaseConstituent(name="chi1", xdo="A ZBY ZZA", u=nc.u_J1, f=nc.f_J1)
_pi1 = BaseConstituent(name="pi1", xdo="A AWZ ZAA", u=nc.u_zero, f=nc.f_unity)
_psi1 = BaseConstituent(name="psi1", xdo="A AAZ ZYY", u=nc.u_zero, f=nc.f_unity)
_phi1 = BaseConstituent(name="phi1", xdo="A ABZ ZZY", u=nc.u_J1, f=nc.f_J1)
_theta1 = BaseConstituent(name="theta1", xdo="A BXA ZZA", u=nc.u_J1, f=nc.f_J1)
_upsilon1 = BaseConstituent(name="upsilon1", xdo="A DZY ZZY", u=nc.u_OO1, f=nc.f_OO1)

# Semi-Diurnals
_2N2 = BaseConstituent(name="2N2", xdo="B XZB ZZZ", u=nc.u_M2, f=nc.f_M2)
_N2 = BaseConstituent(name="N2", xdo="B YZA ZZZ", u=nc.u_M2, f=nc.f_M2)
_nu2 = BaseConstituent(name="nu2", xdo="B YBY ZZZ", u=nc.u_M2, f=nc.f_M2)
_M2 = BaseConstituent(name="M2", xdo="B ZZZ ZZZ", u=nc.u_M2, f=nc.f_M2)
_lambda2 = BaseConstituent(name="lambda2", xdo="B AXA ZZB", u=nc.u_M2, f=nc.f_M2)
_L2 = BaseConstituent(name="L2", xdo="B AZY ZZB", u=nc.u_L2, f=nc.f_L2)
_T2 = BaseConstituent(name="T2", xdo="B BWZ ZAZ", u=nc.u_zero, f=nc.f_unity)
_S2 = BaseConstituent(name="S2", xdo="B BXZ ZZZ", u=nc.u_zero, f=nc.f_unity)
_R2 = BaseConstituent(name="R2", xdo="B BYZ ZYB", u=nc.u_zero, f=nc.f_unity)
_K2 = BaseConstituent(name="K2", xdo="B BZZ ZZZ", u=nc.u_K2, f=nc.f_K2)

# Additional minor semi-diurnals. OQ2 is a lunar elliptic line (M2 node
# factor); H1 and H2 are solar lines requiring no nodal correction.
_OQ2 = BaseConstituent(name="OQ2", xdo="B WZC ZZZ", u=nc.u_M2, f=nc.f_M2)
_H1 = BaseConstituent(name="H1", xdo="B ZYZ ZAZ", u=nc.u_zero, f=nc.f_unity)
_H2 = BaseConstituent(name="H2", xdo="B ZAZ ZYZ", u=nc.u_zero, f=nc.f_unity)

# Third-Diurnals
_M3 = BaseConstituent(
    name="M3", xdo="C ZZZ ZZZ", u=lambda a: nc.u_Modd(a, 3), f=lambda a: nc.f_Modd(a, 3)
)

###### Compound Constituents
# Long Term
_MSF = CompoundConstituent(name="MSF", members=[(_S2, 1), (_M2, -1)])

# Diurnal
_2Q1 = CompoundConstituent(name="2Q1", members=[(_N2, 1), (_J1, -1)])
_rho1 = CompoundConstituent(name="rho1", members=[(_nu2, 1), (_K1, -1)])
_SO1 = CompoundConstituent(name="SO1", members=[(_S2, 1), (_O1, -1)])

# Semi-Diurnal

_mu2 = CompoundConstituent(name="mu2", members=[(_M2, 2), (_S2, -1)])  # 2MS2
_2SM2 = CompoundConstituent(name="2SM2", members=[(_S2, 2), (_M2, -1)])
_epsilon2 = CompoundConstituent(  # MNS2
    name="epsilon2", members=[(_M2, 1), (_N2, 1), (_S2, -1)]
)
_MKS2 = CompoundConstituent(name="MKS2", members=[(_M2, 1), (_K2, 1), (_S2, -1)])
_MSN2 = CompoundConstituent(name="MSN2", members=[(_M2, 1), (_S2, 1), (_N2, -1)])
_eta2 = CompoundConstituent(name="eta2", members=[(_K1, 1), (_J1, 1)])  # KJ2

# Third-Diurnal
_2MK3 = CompoundConstituent(name="2MK3", members=[(_M2, 1), (_O1, 1)])
_MK3 = CompoundConstituent(name="MK3", members=[(_M2, 1), (_K1, 1)])
_MO3 = CompoundConstituent(name="MO3", members=[(_M2, 1), (_O1, 1)])
_SO3 = CompoundConstituent(name="SO3", members=[(_S2, 1), (_O1, 1)])
_SK3 = CompoundConstituent(name="SK3", members=[(_S2, 1), (_K1, 1)])

# Quarter-Diurnal
_MN4 = CompoundConstituent(name="MN4", members=[(_M2, 1), (_N2, 1)])
_M4 = CompoundConstituent(name="M4", members=[(_M2, 2)])
_MS4 = CompoundConstituent(name="MS4", members=[(_M2, 1), (_S2, 1)])
_S4 = CompoundConstituent(name="S4", members=[(_S2, 2)])
_SN4 = CompoundConstituent(name="SN4", members=[(_S2, 1), (_N2, 1)])
_MK4 = CompoundConstituent(name="MK4", members=[(_M2, 1), (_K2, 1)])
_SK4 = CompoundConstituent(name="SK4", members=[(_S2, 1), (_K2, 1)])

# Fifth-Diurnal
_2MK5 = CompoundConstituent(name="2MK5", members=[(_M2, 2), (_K1, 1)])
_2SK5 = CompoundConstituent(name="2SK5", members=[(_S2, 2), (_K1, 1)])

# Sixth-Diurnal
_M6 = CompoundConstituent(name="M6", members=[(_M2, 3)])
_S6 = CompoundConstituent(name="S6", members=[(_S2, 3)])
_2MN6 = CompoundConstituent(name="2MN6", members=[(_M2, 2), (_N2, 1)])
_2MS6 = CompoundConstituent(name="2MS6", members=[(_M2, 2), (_S2, 1)])
_2MK6 = CompoundConstituent(name="2MK6", members=[(_M2, 2), (_K2, 1)])
_2SM6 = CompoundConstituent(name="2SM6", members=[(_M2, 1), (_S2, 2)])
_MSK6 = CompoundConstituent(name="MSK6", members=[(_M2, 1), (_S2, 1), (_K2, 1)])

# Seventh-Diurnal
_3MK7 = CompoundConstituent(name="3MK7", members=[(_M2, 3), (_K1, 1)])

# Eighth-Diurnals
_M8 = CompoundConstituent(name="M8", members=[(_M2, 4)])

# NOAA set (37 tidal constituents)
noaa = [
    _M2,
    _S2,
    _N2,
    _K1,
    _M4,
    _O1,
    _M6,
    _MK3,
    _S4,
    _MN4,
    _nu2,
    _S6,
    _mu2,
    _2N2,
    _OO1,
    _lambda2,
    _S1,
    _M1,
    _J1,
    _Mm,
    _Ssa,
    _Sa,
    _MSF,
    _Mf,
    _rho1,
    _Q1,
    _T2,
    _R2,
    _2Q1,
    _P1,
    _2SM2,
    _M3,
    _L2,
    _2MK3,
    _K2,
    _M8,
    _MS4,
]

# Extended 67-constituent set (the NOAA set above augmented with the additional
# minor and shallow-water constituents), ordered by increasing frequency.
# list provided by Pengcheng Wang, obtained by setting Rayleigh criteria to 0.8
# note M1 in NOAA set is replaced by NO1 here.
extended = [
    _Sa,
    _Ssa,
    _MSm,
    _Mm,
    _MSF,
    _Mf,
    _alpha1,
    _2Q1,
    _sigma1,
    _Q1,
    _rho1,
    _O1,
    _tau1,
    _beta1,
    _NO1,
    _chi1,
    _pi1,
    _P1,
    _S1,
    _K1,
    _psi1,
    _phi1,
    _theta1,
    _J1,
    _SO1,
    _OO1,
    _upsilon1,
    _OQ2,
    _epsilon2,
    _2N2,
    _mu2,
    _N2,
    _nu2,
    _H1,
    _M2,
    _H2,
    _MKS2,
    _lambda2,
    _L2,
    _T2,
    _S2,
    _R2,
    _K2,
    _MSN2,
    _eta2,
    _MO3,
    _M3,
    _SO3,
    _MK3,
    _SK3,
    _MN4,
    _M4,
    _SN4,
    _MS4,
    _MK4,
    _S4,
    _SK4,
    _2MK5,
    _2SK5,
    _2MN6,
    _M6,
    _2MS6,
    _2MK6,
    _2SM6,
    _MSK6,
    _3MK7,
    _M8,
]
