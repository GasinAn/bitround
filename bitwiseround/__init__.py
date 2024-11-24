"""
Bitwiseround
========

Bitwise rounding for fast low-rate lossy compression

Overview
--------

The Numpy ufunc `bitwiseround.bitwise_round` makes bitwise rounding for floats.
It takes two inputs: `a` and `d`. It modifies `a` to ``sign(a) * round(abs(a)/
2**n) * 2**n``, where ``n == floor(log2(d)) + 1``. After bitwise rounding,
elements in `a` will have more bits with the value of 0 at the ends. This makes
`bitwiseround.bitwise_round` cooperate with Bitshuffle compression compression
well. It will make the compression rate lower, while leading to a loss of
precision controlled by `d`.

Warning
-------

At this moment, `bitwiseround.bitwise_round` works only for float64s.

The behavior of `bitwiseround.bitwise_round` is UNDEFINED when one of the
following situations holds:

 * `d` is not a positive normal number;
 * `a` is a subnormal number;
 * ``floor(log2(a)) - floor(log2(d)) <= 1022``.
"""

from ._core import bitwise_round
from ._test import test

__all__ = ['bitwise_round', 'test']
