from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension('conv_utils', ['conv_utils.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
)
