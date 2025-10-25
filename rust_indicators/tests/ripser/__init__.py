"""
Ripser 测试套件

这个测试包包含 Ripser Rust 实现的所有测试，
每个迭代都会添加相应的测试模块。

测试策略：
1. 使用 giotto-ph 生成参考数据
2. 对比 Rust 实现与参考数据的数值差异
3. 验证误差在容忍范围内（< 1e-6）
4. 性能基准测试（vs giotto-ph）

测试模块：
- test_binomial.py: 二项式系数表测试（迭代1）
- test_distance_matrix.py: 距离矩阵测试（迭代2）
- test_simplex.py: 简单复形测试（迭代3-4）
- test_cohomology.py: 上同调计算测试（迭代5）
- test_integration.py: 端到端集成测试（迭代6）
"""
