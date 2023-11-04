import setuptools
import numpy

setuptools.setup(
    name='bitround',
    version='0.4.2',
    description='"Bit round" for fast lossy compression',
    author='GasinAn',
    author_email='Gasin185@163.com',
    maintainer='GasinAn',
    maintainer_email='Gasin185@163.com',
    url='https://github.com/GasinAn/bitround',
    license='MIT',
    python_requires='>=3.0',
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
