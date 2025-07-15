from setuptools import setup, Extension
import numpy

# 模块名称
m_name = "grid_subsampling"

# C++源码路径
SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "grid_subsampling/grid_subsampling.cpp",
    "wrapper.cpp"
]

# 创建扩展模块
module = Extension(
    m_name,
    sources=SOURCES,
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0']
)

# 构建
setup(
    name=m_name,
    ext_modules=[module]
)
