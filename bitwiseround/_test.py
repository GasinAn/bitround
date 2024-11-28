from ._core import bitwise_round
import numpy as np


@np.vectorize
def _bitwise_round(a, d):
    """A function which should be equivalent to bitwiseround.bitwise_round"""

    if np.isfinite(a):
        exp2_n = np.exp2(np.floor(np.log2(d))+1)
        return np.sign(a)*np.round(np.abs(a)/exp2_n)*exp2_n
    else:
        return a


def test():
    """Test for bitwiseround.bitwise_round"""

    a = np.array([-0.0, +0.0])
    d = np.exp2(np.random.uniform(-1, 1))
    bitwise_round(a, d)
    assert np.all(a == 0) and np.all(np.signbit(a) == np.array([1, 0]))

    a = np.array([-np.inf, +np.inf])
    d = np.exp2(np.random.uniform(-1, 1))
    bitwise_round(a, d)
    assert np.all(np.isinf(a)) and np.all(np.signbit(a) == np.array([1, 0]))

    a = np.array([-np.nan, +np.nan])
    d = np.exp2(np.random.uniform(-1, 1))
    bitwise_round(a, d)
    assert np.all(np.isnan(a)) and np.all(np.signbit(a) == np.array([1, 0]))

    a = np.logspace(-64, +64, 3**12, base=2)
    d = np.exp2(np.random.uniform(-1, 1))
    _a = _bitwise_round(a, d)
    bitwise_round(a, d)
    assert np.all(a == _a) and np.all(np.signbit(a) == np.signbit(_a))

    print("Test passed.")
