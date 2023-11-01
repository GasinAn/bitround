from setuptools import Extension
from setuptools import setup

setup(
    name="bitround",
    version="0.3.0",
    author="GasinAn",
    author_email="Gasin185@163.com",
    maintainer="GasinAn",
    maintainer_email="Gasin185@163.com",
    url="https://github.com/GasinAn/bitround",
    license="MIT",
    packages=["bitround"],
    package_dir={"bitround": "bitround"},
    ext_modules=[
        Extension(
            name="bitround.core",
            sources=["bitround/core.c"],
        )
    ],
    py_modules=["bitround.test"],
    python_requires=">=3.0",
    install_requires=[],
)
