from ._core import bitround
from numpy import *

@vectorize
def _bitround(a, d):
    if (a==0) or isinf(a) or isnan(a): return a
    n = floor(log2(d))
    return sign(a)*floor((abs(a)+exp2(n))/exp2(n+1))*exp2(n+1)

def test():
    print('func test')
