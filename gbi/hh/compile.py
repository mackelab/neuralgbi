import numpy as np

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "biophys_cython_comp",
        ["biophys_cython_comp.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(extensions))
