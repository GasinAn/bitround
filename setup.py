import setuptools
import numpy

setuptools.setup(
    packages=['bitwiseround'],
    package_dir={'bitwiseround': 'bitwiseround'},
    ext_modules=[
        setuptools.Extension(
            name='bitwiseround._core',
            sources=['bitwiseround/_core.c'],
            include_dirs=[numpy.get_include()],
        )
    ],
    py_modules=['bitwiseround._test'],
)
