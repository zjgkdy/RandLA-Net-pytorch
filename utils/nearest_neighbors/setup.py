from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "nearest_neighbors",
        sources=["knn.pyx", "knn_.cxx"],
        language="c++",
        include_dirs=["./", numpy.get_include()],
        extra_compile_args=["-std=c++11", "-fopenmp"],
        extra_link_args=["-std=c++11", "-fopenmp"]
    )
]

setup(
    name="KNN_NanoFLANN",
    ext_modules=cythonize(ext_modules),
)
