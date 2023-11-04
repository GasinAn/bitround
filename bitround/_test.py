from ._core import bitround
from numpy import *

@vectorize
def _bitround(a, d):
    if (a==0) or isinf(a) or isnan(a): return a
    exp2_n = exp2(ceil(log2(d)))
    return sign(a)*round(abs(a)/exp2_n)*exp2_n

def test():
    print('func test')
