# Bitwiseround

Bitwise rounding for fast low-rate lossy compression

## Overview

The Numpy ufunc `bitwiseround.bitwise_round` makes bitwise rounding for floats. It takes two inputs: $a$ and $d$. It modifies $a$ to $\text{sgn}(a) \cdot \text{round}(\text{abs}(a)/2^{n}) \cdot 2^{n}$, where $n = \text{floor}(\log_{2}(d)) + 1$. After bitwise rounding, elements in $a$ will have more bits with the value of $0$ at the ends. This makes `bitwiseround.bitwise_round` cooperate with [Bitshuffle](https://github.com/kiyo-masui/bitshuffle) compression well. It will make the compression rate lower, while leading to a loss of precision controlled by $d$.

## Warning

At this moment, `bitwiseround.bitwise_round` works only for float64s.

The behavior of `bitwiseround.bitwise_round` is UNDEFINED when one of the following situations holds:

 * $d$ is not a positive normal number;
 * $a$ is a subnormal number;
 * $\text{floor}(\log_{2}(a)) - \text{floor}(\log_{2}(d)) \leq 1022$.

## Installation

### Building from Source

[build](https://pypa-build.readthedocs.io/en/stable/), [Setuptools](https://setuptools.pypa.io/en/latest/), [Numpy](https://numpy.org/) and a C compiler (such as [GCC](https://gcc.gnu.org/)) should be installed first.

```
git clone https://github.com/GasinAn/bitwiseround.git
cd bitwiseround
python -m build -n
```

Then the package can be installed by using [pip](https://pip.pypa.io/en/stable/):

```
pip install dist/*.whl
```

Now LEAVE the `bitwiseround` directory to use the package.

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
