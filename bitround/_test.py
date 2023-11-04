from ._core import bitround
from numpy import *

@vectorize
def _bitround(a, d):
    if isfinite(a):
        exp2_n = exp2(ceil(log2(d)))
        return sign(a)*round(abs(a)/exp2_n)*exp2_n
    else:
        return a

def test():
    print('func test')
