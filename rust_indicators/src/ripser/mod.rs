/*!
 * Ripser - 持久同调计算的 Rust 实现
 *
 * 这是 ripser (https://github.com/Ripser/ripser) 的完整 Rust 重写版本，
 * 专为 jesse-trade 量化交易框架中的拓扑数据分析优化。
 *
 * # 算法概述
 *
 * Ripser 实现了 Vietoris-Rips 持久同调的高效计算：
 * 1. 从点云或距离矩阵构建 Vietoris-Rips 复形
 * 2. 使用矩阵归约算法计算持久同调
 * 3. 输出持久性图（persistence diagram）
 *
 * # 核心特性
 *
 * - **组合编码**: 使用二项式系数将 k-simplex 编码为单个整数索引
 * - **矩阵归约**: 基于 pivot 的上同调计算
 * - **并行计算**: 高维同调的多线程并行化
 * - **内存优化**: 压缩存储距离矩阵，减少内存占用
 *
 * # 模块结构
 *
 * - `types`: 核心数据类型定义
 * - `core`: 核心算法实现
 *   - `binomial`: 二项式系数表
 *   - `distance`: 距离矩阵（稠密/稀疏）
 *   - `simplex`: 简单复形编解码
 *   - `cohomology`: 上同调计算
 *   - `barcode`: 持久性条形码生成
 * - `parallel`: 并行化实现（后期）
 * - `ffi`: Python 绑定（最后阶段）
 *
 * # 开发状态
 *
 * 当前处于迭代 0：研究与架构设计阶段
 *
 * # 参考文献
 *
 * - [Ripser](https://github.com/Ripser/ripser): Ulrich Bauer 的原始 C++ 实现
 * - [giotto-ph](https://github.com/giotto-ai/giotto-ph): Python 包装和扩展
 * - [Computational Topology for Data Analysis](https://www.cs.duke.edu/courses/fall06/cps296.1/)
 */

// 子模块声明
pub mod types;
pub mod core;
pub mod ffi;

// 重导出核心类型（便于使用）
pub use types::*;
pub use ffi::*;

// 版本信息
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_loads() {
        // 基础冒烟测试：模块可以正常加载
        assert!(!VERSION.is_empty());
    }
}
