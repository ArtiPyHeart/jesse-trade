#!/bin/bash

cd custom_indicators/toolbox/feature_selection

# 清理之前的编译文件
rm -rf build/
rm -f *.so
rm -f *.c
rm -f *.html

# 设置环境变量以启用更多优化
export CFLAGS="-O3 -march=native -ffast-math"

# 编译
python setup_fcq_cython.py build_ext --inplace

# 如果生成了注释文件，将其移动到docs目录
if [ -f fcq_selector_cython.html ]; then
    mkdir -p docs
    mv fcq_selector_cython.html docs/
fi

