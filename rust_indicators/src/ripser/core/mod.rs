/*!
 * 核心算法模块
 *
 * 这个模块包含 Ripser 算法的所有核心组件。
 * 每个子模块负责一个独立的功能，按照迭代计划逐步实现。
 *
 * # 子模块
 *
 * ## 已实现
 *
 * （迭代 0：暂无）
 *
 * ## 计划实现
 *
 * - `binomial`（迭代 1）: 二项式系数表
 * - `distance`（迭代 2）: 距离矩阵（稠密/稀疏）
 * - `simplex`（迭代 3-4）: 简单复形编解码与边枚举
 * - `cohomology`（迭代 5）: 上同调计算（矩阵归约）
 * - `barcode`（迭代 6）: 持久性条形码生成
 *
 * # 实现顺序
 *
 * 模块间存在依赖关系，必须按以下顺序实现：
 *
 * ```text
 * binomial
 *    ↓
 * simplex ← distance
 *    ↓
 * cohomology
 *    ↓
 * barcode
 * ```
 */

// 子模块声明（按迭代顺序启用）

// 迭代 1: 二项式系数表
pub mod binomial;

// 迭代 2: 距离矩阵
pub mod distance;

// 迭代 3-4: 简单复形
pub mod simplex;

// 迭代 5: 上同调计算
pub mod cohomology;

// 迭代 6: 端到端算法
pub mod algorithm;

// 未来: 持久性条形码
// pub mod barcode;

// 重导出（便于外部使用）
pub use binomial::*;
pub use distance::*;
pub use simplex::*;
pub use cohomology::*;
pub use algorithm::*;
// pub use barcode::*;
