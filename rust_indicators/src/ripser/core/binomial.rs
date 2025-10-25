/*!
 * 二项式系数表（Binomial Coefficient Table）
 *
 * 这个模块实现了用于 Simplex 组合编码的二项式系数表。
 *
 * # 核心概念
 *
 * 二项式系数 C(n, k) = n! / (k! * (n-k)!)，也写作 "n choose k"。
 * 在 Ripser 中用于将 k-simplex 的顶点集合编码为单个整数索引。
 *
 * # 编码方案
 *
 * 对于 k-simplex {v0, v1, ..., vk}（其中 v0 < v1 < ... < vk）：
 *
 * ```text
 * index = C(vk, k+1) + C(vk-1, k) + ... + C(v1, 2) + v0
 * ```
 *
 * # 优化策略
 *
 * 1. **预计算**: 构建时一次性计算所有需要的系数
 * 2. **动态规划**: 使用 Pascal's triangle 递推公式
 * 3. **转置存储**: B[k][n] 而非 B[n][k]（行访问模式）
 * 4. **溢出检测**: 检查索引是否超出 i64 范围
 *
 * # 示例
 *
 * ```ignore
 * use ripser::core::binomial::BinomialCoeffTable;
 *
 * let table = BinomialCoeffTable::new(10, 3);
 *
 * // C(5, 2) = 10
 * assert_eq!(table.get(5, 2), 10);
 *
 * // C(6, 3) = 20
 * assert_eq!(table.get(6, 3), 20);
 * ```
 */

use crate::ripser::types::Index;
use std::fmt;

/// 二项式系数表
///
/// 存储所有 C(i, j) 的值，其中 0 ≤ i ≤ n 且 0 ≤ j ≤ k。
///
/// # 内存布局
///
/// 使用转置存储：`table[j][i] = C(i, j)`
///
/// 这样做是因为访问模式通常是固定 k 遍历不同的 n，
/// 转置后可以获得更好的缓存局部性。
///
/// # 复杂度
///
/// - 构建时间: O(n * k)
/// - 构建空间: O(n * k)
/// - 查询时间: O(1)
///
#[derive(Clone, Debug)]
pub struct BinomialCoeffTable {
    /// 二项式系数表
    /// table[k][n] = C(n, k)
    table: Vec<Vec<Index>>,

    /// 最大 n 值
    max_n: usize,

    /// 最大 k 值
    max_k: usize,
}

impl BinomialCoeffTable {
    /// 创建新的二项式系数表
    ///
    /// # 参数
    ///
    /// - `n`: 最大顶点索引（通常是点云大小）
    /// - `k`: 最大维度（通常是 max_dim + 2）
    ///
    /// # 返回
    ///
    /// 预计算好的二项式系数表
    ///
    /// # Panics
    ///
    /// 如果计算过程中发生索引溢出（超出 i64::MAX）
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let table = BinomialCoeffTable::new(100, 5);
    /// assert_eq!(table.get(10, 3), 120);
    /// ```
    pub fn new(n: usize, k: usize) -> Self {
        // 预分配矩阵：table[k+1][n+1]
        let mut table = vec![vec![0i64; n + 1]; k + 1];

        // 使用 Pascal's triangle 递推公式构建
        // C(n, k) = C(n-1, k-1) + C(n-1, k)
        // 边界条件：C(n, 0) = 1, C(n, n) = 1

        for i in 0..=n {
            // C(i, 0) = 1（从 i 中选 0 个）
            table[0][i] = 1;

            // C(i, i) = 1（从 i 中选 i 个）
            if i <= k {
                table[i][i] = 1;
            }

            // 递推计算 C(i, j)
            for j in 1..std::cmp::min(i, k + 1) {
                table[j][i] = table[j - 1][i - 1] + table[j][i - 1];

                // 溢出检测（在关键位置检查）
                // 检查 C(i, i/2) 通常是最大值
                if j == i / 2 || j == (i + 1) / 2 {
                    Self::check_overflow(table[j][i], i, j);
                }
            }
        }

        Self {
            table,
            max_n: n,
            max_k: k,
        }
    }

    /// 获取二项式系数 C(n, k)
    ///
    /// # 参数
    ///
    /// - `n`: 集合大小
    /// - `k`: 选择数量
    ///
    /// # 返回
    ///
    /// C(n, k) 的值
    ///
    /// # Panics
    ///
    /// 如果 `n > max_n` 或 `k > max_k` 或 `k > n`
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let table = BinomialCoeffTable::new(10, 5);
    /// assert_eq!(table.get(5, 2), 10);  // C(5, 2) = 10
    /// assert_eq!(table.get(6, 3), 20);  // C(6, 3) = 20
    /// ```
    #[inline]
    pub fn get(&self, n: usize, k: usize) -> Index {
        // 数学上，C(n, k) = 0 when k > n
        if k > n {
            return 0;
        }

        // 边界检查
        assert!(
            k < self.table.len(),
            "k={} exceeds max_k={}",
            k,
            self.max_k
        );
        assert!(
            n < self.table[k].len(),
            "n={} exceeds max_n={}",
            n,
            self.max_n
        );

        self.table[k][n]
    }

    /// 获取最大支持的 n 值
    #[inline]
    pub fn max_n(&self) -> usize {
        self.max_n
    }

    /// 获取最大支持的 k 值
    #[inline]
    pub fn max_k(&self) -> usize {
        self.max_k
    }

    /// 检查索引溢出
    ///
    /// 在 Ripser 中，simplex 索引必须 < 2^55（为系数预留 8 位）
    fn check_overflow(value: Index, n: usize, k: usize) {
        // 最大安全索引（为系数域预留 8 位）
        const NUM_COEFFICIENT_BITS: usize = 8;
        const MAX_SIMPLEX_INDEX: i64 =
            (1i64 << (64 - 1 - NUM_COEFFICIENT_BITS)) - 1;

        if value > MAX_SIMPLEX_INDEX {
            panic!(
                "Binomial coefficient overflow: C({}, {}) = {} > max_index = {}",
                n, k, value, MAX_SIMPLEX_INDEX
            );
        }

        // 额外检查负值（虽然理论上不会发生）
        if value < 0 {
            panic!(
                "Binomial coefficient underflow: C({}, {}) = {}",
                n, k, value
            );
        }
    }
}

impl fmt::Display for BinomialCoeffTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BinomialCoeffTable(max_n={}, max_k={})",
            self.max_n, self.max_k
        )
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_basic() {
        let table = BinomialCoeffTable::new(10, 5);

        // C(0, 0) = 1
        assert_eq!(table.get(0, 0), 1);

        // C(5, 0) = 1
        assert_eq!(table.get(5, 0), 1);

        // C(5, 5) = 1
        assert_eq!(table.get(5, 5), 1);

        // C(5, 1) = 5
        assert_eq!(table.get(5, 1), 5);

        // C(5, 2) = 10
        assert_eq!(table.get(5, 2), 10);

        // C(5, 3) = 10
        assert_eq!(table.get(5, 3), 10);

        // C(5, 4) = 5
        assert_eq!(table.get(5, 4), 5);
    }

    #[test]
    fn test_binomial_pascal_triangle() {
        // 验证 Pascal's triangle 的几行
        let table = BinomialCoeffTable::new(10, 5);

        // 第 0 行: 1
        assert_eq!(table.get(0, 0), 1);

        // 第 1 行: 1 1
        assert_eq!(table.get(1, 0), 1);
        assert_eq!(table.get(1, 1), 1);

        // 第 2 行: 1 2 1
        assert_eq!(table.get(2, 0), 1);
        assert_eq!(table.get(2, 1), 2);
        assert_eq!(table.get(2, 2), 1);

        // 第 3 行: 1 3 3 1
        assert_eq!(table.get(3, 0), 1);
        assert_eq!(table.get(3, 1), 3);
        assert_eq!(table.get(3, 2), 3);
        assert_eq!(table.get(3, 3), 1);

        // 第 4 行: 1 4 6 4 1
        assert_eq!(table.get(4, 0), 1);
        assert_eq!(table.get(4, 1), 4);
        assert_eq!(table.get(4, 2), 6);
        assert_eq!(table.get(4, 3), 4);
        assert_eq!(table.get(4, 4), 1);
    }

    #[test]
    fn test_binomial_well_known_values() {
        let table = BinomialCoeffTable::new(20, 10);

        // C(10, 5) = 252
        assert_eq!(table.get(10, 5), 252);

        // C(10, 3) = 120
        assert_eq!(table.get(10, 3), 120);

        // C(20, 10) = 184756
        assert_eq!(table.get(20, 10), 184756);

        // C(15, 7) = 6435
        assert_eq!(table.get(15, 7), 6435);
    }

    #[test]
    fn test_binomial_symmetry() {
        // 验证对称性: C(n, k) = C(n, n-k)
        let table = BinomialCoeffTable::new(10, 10);

        for n in 0..=10 {
            for k in 0..=n {
                assert_eq!(
                    table.get(n, k),
                    table.get(n, n - k),
                    "Symmetry failed: C({}, {}) != C({}, {})",
                    n,
                    k,
                    n,
                    n - k
                );
            }
        }
    }

    #[test]
    fn test_binomial_recursive_property() {
        // 验证递推性质: C(n, k) = C(n-1, k-1) + C(n-1, k)
        let table = BinomialCoeffTable::new(10, 5);

        for n in 2..=10 {
            // 从 n=2 开始，避免访问 C(0, 1)
            for k in 1..std::cmp::min(n, 5) {
                // k < n 避免边界问题
                let expected = table.get(n - 1, k - 1) + table.get(n - 1, k);
                assert_eq!(
                    table.get(n, k),
                    expected,
                    "Recursive property failed at C({}, {})",
                    n,
                    k
                );
            }
        }
    }

    #[test]
    fn test_binomial_large_values() {
        // 测试较大的值（但不溢出）
        let table = BinomialCoeffTable::new(50, 25);

        // C(50, 25) 是一个非常大的数，但应该在 i64 范围内
        let c_50_25 = table.get(50, 25);
        assert!(c_50_25 > 0);
        assert!(c_50_25 < i64::MAX);

        // C(40, 20) = 137846528820
        assert_eq!(table.get(40, 20), 137846528820);
    }

    #[test]
    #[should_panic(expected = "exceeds max_n")]
    fn test_binomial_out_of_bounds_n() {
        let table = BinomialCoeffTable::new(10, 5);
        let _ = table.get(11, 2); // n > max_n
    }

    #[test]
    #[should_panic(expected = "exceeds max_k")]
    fn test_binomial_out_of_bounds_k() {
        let table = BinomialCoeffTable::new(10, 5);
        let _ = table.get(8, 6); // k > max_k
    }

    #[test]
    fn test_binomial_k_greater_than_n() {
        let table = BinomialCoeffTable::new(10, 5);
        // 数学上，C(n, k) = 0 when k > n
        assert_eq!(table.get(3, 5), 0, "C(3, 5) should be 0");
        assert_eq!(table.get(1, 2), 0, "C(1, 2) should be 0");
        assert_eq!(table.get(0, 1), 0, "C(0, 1) should be 0");
    }

    #[test]
    fn test_binomial_edge_cases() {
        let table = BinomialCoeffTable::new(5, 5);

        // C(0, 0) = 1
        assert_eq!(table.get(0, 0), 1);

        // C(n, 0) = 1 对所有 n
        for n in 0..=5 {
            assert_eq!(table.get(n, 0), 1);
        }

        // C(n, n) = 1 对所有 n
        for n in 0..=5 {
            assert_eq!(table.get(n, n), 1);
        }

        // C(n, 1) = n 对所有 n
        for n in 1..=5 {
            assert_eq!(table.get(n, 1), n as i64);
        }
    }

    #[test]
    fn test_binomial_display() {
        let table = BinomialCoeffTable::new(100, 10);
        let display = format!("{}", table);
        assert!(display.contains("max_n=100"));
        assert!(display.contains("max_k=10"));
    }
}
