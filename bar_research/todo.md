# 目标

基于gplearn版本的符号回归，实现一个DEAP版本的符号回归，来实现寻找kurtosis最小的自定义bar。在此基础上，在DEAP中实现尽可能好的符号回归算法。可以通过参考论文和其他符号回归库来实现。

# 实现要领

- 注意gplearn中的自定义loss函数含义，目标是评估整合后的bar的kurtosis，需要在deap中实现类似的逻辑
- DEAP的实现应当包含详尽的注释，方便后续的阅读和修改
- 如果你希望进行算法测试，请使用data/btc_1m.npy的最后1万行数据作为测试数据
- gplearn的符号回归算法并不一定是最好的，因此我非常推荐你通过参考其他符号回归库，实现一个更好的符号回归算法，比如：
    - 使用julia语言的PySR库[https://github.com/MilesCranmer/PySR]

# 实现进度

## 已完成

1. **基础DEAP符号回归实现** (`symbolic_regression_deap.py`)
   - 实现了与gplearn类似的峰度最小化目标
   - 支持自定义适应度函数
   - 包含基本的遗传操作（交叉、变异、选择）
   - 添加了Hall of Fame保存最佳个体
   - 支持并行计算

2. **高级DEAP符号回归实现** (`symbolic_regression_deap_advanced.py`)
   - **多目标优化**: 使用NSGA-II同时优化峰度和复杂度
   - **岛屿模型**: 支持并行进化多个种群，增加多样性
   - **语义交叉**: 基于语义相似度的智能交叉操作
   - **局部搜索**: 对常数进行微调优化
   - **自适应变异**: 根据进化进度和停滞情况调整变异率
   - **高级数学操作符**: 包括条件运算、保护函数等
   - **Pareto前沿可视化**: 展示多目标优化结果
   - **模型持久化**: 支持保存和加载训练好的模型

3. **测试脚本** (`test_symbolic_regression.py`)
   - 对比基础版本和高级版本的性能
   - 可视化生成的bar分布
   - 统计分析（峰度、偏度、正态性检验）

# 使用方法

## 基础版本快速开始
```python
from symbolic_regression_deap import SymbolicRegressionDEAP, prepare_features

# 准备数据
X, feature_names, _ = prepare_features(use_last_n=10000)

# 创建并训练模型
model = SymbolicRegressionDEAP(
    population_size=200,
    generations=50,
    verbose=True
)
model.fit(X, feature_names)

# 获取结果
results = model.evaluate_best_individual()
print(f"最佳表达式: {results['expression']}")
print(f"峰度: {results['kurtosis']}")
```

## 高级版本使用
```python
from symbolic_regression_deap_advanced import AdvancedSymbolicRegressionDEAP

# 创建多目标优化模型
model = AdvancedSymbolicRegressionDEAP(
    population_size=400,
    generations=100,
    n_islands=4,  # 使用4个岛屿并行进化
    adaptive_mutation=True,  # 自适应变异率
    verbose=True
)

# 训练
model.fit(X, feature_names)

# 获取Pareto最优解
best_solutions = model.get_best_expressions(n=5)

# 可视化Pareto前沿
model.visualize_pareto_front()
```

# 参考文献

1. **DEAP框架**
   - Fortin, F. A., et al. (2012). "DEAP: Evolutionary algorithms made easy." Journal of Machine Learning Research, 13(Jul), 2171-2175.

2. **多目标优化**
   - Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on evolutionary computation, 6(2), 182-197.

3. **语义遗传编程**
   - Moraglio, A., et al. (2012). "Geometric semantic genetic programming." International Conference on Parallel Problem Solving from Nature.

4. **岛屿模型**
   - Whitley, D., et al. (1999). "The island model genetic algorithm: On separability, population size and convergence." Journal of computing and information technology, 7(1), 33-47.

5. **符号回归综述**
   - Koza, J. R. (1992). "Genetic programming: on the programming of computers by means of natural selection." MIT press.
   - Schmidt, M., & Lipson, H. (2009). "Distilling free-form natural laws from experimental data." Science, 324(5923), 81-85.