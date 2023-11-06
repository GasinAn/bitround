import setuptools
import numpy

setuptools.setup(
    packages=['bitround'],
    package_dir={'bitround': 'bitround'},
    ext_modules=[
        setuptools.Extension(
            name='bitround._core',
            sources=['bitround/_core.c'],
            include_dirs=[numpy.get_include()],
        )
    ],
    py_modules=['bitround._test'],
)
