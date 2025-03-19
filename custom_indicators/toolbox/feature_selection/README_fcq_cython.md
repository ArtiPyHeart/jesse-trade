# Cython版FCQSelector

这是一个使用Cython优化的FCQSelector实现，用于特征选择任务。通过将关键计算部分用Cython重写，可以显著提高特征选择的性能，特别是在处理大型数据集时。

## 安装依赖

在使用之前，确保已安装以下依赖项：

```bash
pip install numpy pandas scikit-learn tqdm cython
```

## 编译Cython代码

在使用之前，需要先编译Cython代码。在当前目录下运行：

```bash
cd custom_indicators
python setup_fcq_cython.py build_ext --inplace
```

编译成功后，会在当前目录下生成相应的动态链接库文件（`.so`或`.pyd`）。

## 使用方法

编译成功后，可以像使用普通Python模块一样导入并使用：

```python
from custom_indicators.toolbox.feature_selction.fcq_selector_cython import CythonFCQSelector

# 创建选择器实例
selector = CythonFCQSelector(max_features=20, regression=False)

# 拟合数据并转换
X_selected = selector.fit_transform(X, y)

# 或者分步进行
selector.fit(X, y)
X_selected = selector.transform(X)

# 获取选中的特征掩码
mask = selector.get_support()
```

## 性能对比

使用`fcq_cython_example.py`脚本可以比较原始FCQSelector和Cython版本的性能差异：

```bash
python fcq_cython_example.py
```

通常，Cython版本在处理大型数据集时能够提供2-5倍的性能提升，同时保持相同的特征选择结果。

## 优化要点

1. 使用Cython的静态类型声明
2. 使用C库中的数学函数
3. 关闭边界检查和负索引检查
4. 启用C除法
5. 为循环预先分配内存和类型
6. 避免Python对象的创建和销毁

## 注意事项

1. 使用前需要确保已经编译Cython代码
2. 输入数据必须是pandas DataFrame和Series
3. 结果与原始版本FCQSelector完全一致，但速度显著提升