# Bitwiseround

Bitwise rounding for fast low-rate lossy compression

## Overview

The Numpy ufunc `bitwiseround.bitwise_round` makes bitwise rounding for float. It takes two input: $a$ and $d$. It modifies $a$ to $\text{sgn}(a)\,\text{round}(\text{abs}(a)/2^{n})\,2^{n}$, where $n = \text{floor}(\log_{2}(d))+1$. After bitwise rounding, elements in $a$ will have more `0` tail bits. This makes `bitwiseround.bitwise_round` cooperate with [Bitshuffle](https://github.com/kiyo-masui/bitshuffle) compression well. It will makes the compression rate lower, while leaves loss of precision controlled by $d$.

## Warning

At this moment, `bitwiseround.bitwise_round` works only for float64.

The behavior of `bitwiseround.bitwise_round` is UNDEFINED when it is one of following situations:

 * $d$ is not positive normal number;
 * $a$ is subnormal number;
 * $\text{floor}(\log_{2}(a))-\text{floor}(\log_{2}(d))\leq1022$.

## Installation

### Building from Source

[build](https://pypa-build.readthedocs.io/en/stable/), [Numpy](https://numpy.org/), [Setuptools](https://setuptools.pypa.io/en/latest/) and a C compiler (such as [GCC](https://gcc.gnu.org/)) should be installed first.

```
git clone https://github.com/GasinAn/bitwiseround.git
cd bitwiseround
python -m build -n
```

Then the package can be installed by using pip:

```
pip install dist/*.whl
```

## Example

```python
import numpy
import h5py
import bitshuffle.h5

from bitwiseround import bitwise_round

file = h5py.File('example.hdf5', 'w')

dataset = file.create_dataset(
    'data',
    (100, 100, 100),
    compression=bitshuffle.h5.H5FILTER,
    compression_opts=(0, bitshuffle.h5.H5_COMPRESS_LZ4),
    dtype='float64',
    )

data = numpy.random.rand(100, 100, 100)
data = data.astype('float64')

# Use bitwise_round here!
bitwise_round(data, 0.333)

dataset[...] = data

file.close()
```

## See also

 * [Bitshuffle](https://github.com/kiyo-masui/bitshuffle)
