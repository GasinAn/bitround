"""
Bitround
========

"Bit round" for fast low-rate lossy compression

Overview
--------

The Numpy ufunc bitround.bitround makes "bit round" for float. It takes two
input: a and d. It modifies a to sign(a)round(abs(a)/2**n)2**n, where n == floor
(log2(d))+1. After "bit round", elements in a will have more 0 tail bits. This
makes bitround.bitround cooperate with Bitshuffle compression well. It will
makes the compression rate lower, while leaves loss of precision controlled by
d.

Warning
-------

At this moment, bitround.bitround works only for float64.

The behavior of bitround.bitround is UNDEFINED when it is one of following
situations:

 * d is not positive normal number;
 * a is subnormal number;
 * floor(log2(a)) - floor(log2(d)) <= 1022.
"""

from ._core import bitwise_round
from ._test import test

__all__ = ['bitwise_round', 'test']
