import math

from numba import njit


@njit
def sin_radians(degrees: float) -> float:
    return math.sin(math.radians(degrees))


@njit
def cos_radians(degrees: float) -> float:
    return math.cos(math.radians(degrees))


@njit
def tan_radians(degrees: float) -> float:
    return math.tan(math.radians(degrees))
