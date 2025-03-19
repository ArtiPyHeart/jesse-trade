"""
编译Cython版本的FCQSelector的setup脚本
运行方式: python setup_fcq_cython.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# 设置编译器优化标志
extra_compile_args = [
    "-O3",  # 最高级别的优化
    "-march=native",  # 针对本机CPU架构优化
    "-ffast-math",  # 启用快速数学运算
    "-ftree-vectorize",  # 启用向量化
]

extensions = [
    Extension(
        "fcq_selector_cython",
        ["fcq_selector_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_TRACE", "0"),  # 禁用跟踪
            ("CYTHON_PROFILE", "0"),  # 禁用性能分析
        ],
    )
]

setup(
    name="fcq_selector_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,  # 禁用边界检查
            "wraparound": False,  # 禁用负索引
            "cdivision": True,  # 禁用除零检查
            "initializedcheck": False,  # 禁用初始化检查
            "nonecheck": False,  # 禁用None检查
            "overflowcheck": False,  # 禁用溢出检查
            "profile": False,  # 禁用性能分析
            "linetrace": False,  # 禁用行跟踪
        },
        annotate=True,  # 生成注释HTML文件以查看C代码转换
    ),
)
