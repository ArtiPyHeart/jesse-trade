/*!
 * 距离矩阵实现
 *
 * 这个模块实现了两种距离矩阵：
 * 1. `CompressedDistanceMatrix`: 稠密矩阵（压缩存储，仅存储下三角）
 * 2. `SparseDistanceMatrix`: 稀疏矩阵（邻接表存储）
 *
 * # 设计原则
 *
 * - **内存效率**: 稠密矩阵仅存储 n*(n-1)/2 个元素
 * - **访问速度**: O(1) 查询复杂度
 * - **数值精度**: 使用 f32 与 C++ ripser 保持一致
 *
 * # 使用示例
 *
 * ```ignore
 * // 从点云构建距离矩阵
 * let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
 * let dist = CompressedDistanceMatrix::from_points(&points, Metric::Euclidean);
 *
 * // 查询距离
 * let d01 = dist.get(0, 1);  // 点0和点1之间的距离
 * ```
 */

use super::super::types::Value;
use std::fmt;

// ============================================================================
// Metric - 距离度量
// ============================================================================

/// 距离度量类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// 欧几里得距离（L2范数）
    Euclidean,
    /// 曼哈顿距离（L1范数）
    Manhattan,
    /// 最大范数（L∞范数）
    Chebyshev,
}

impl Metric {
    /// 计算两点之间的距离
    #[inline]
    pub fn distance(&self, p1: &[f32], p2: &[f32]) -> f32 {
        assert_eq!(
            p1.len(),
            p2.len(),
            "Points must have same dimension: {} vs {}",
            p1.len(),
            p2.len()
        );

        match self {
            Metric::Euclidean => {
                let sum: f32 = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum();
                sum.sqrt()
            }
            Metric::Manhattan => p1
                .iter()
                .zip(p2.iter())
                .map(|(x, y)| (x - y).abs())
                .sum(),
            Metric::Chebyshev => p1
                .iter()
                .zip(p2.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max),
        }
    }
}

// ============================================================================
// CompressedDistanceMatrix - 稠密矩阵（压缩存储）
// ============================================================================

/// 压缩距离矩阵（仅存储下三角）
///
/// # 内存布局
///
/// 对于 n 个点的距离矩阵，仅存储下三角部分（不包括对角线）：
/// ```text
///   0   1   2   3
/// 0 0   -   -   -
/// 1 d10 0   -   -
/// 2 d20 d21 0   -
/// 3 d30 d31 d32 0
/// ```
///
/// 存储为一维数组: [d10, d20, d21, d30, d31, d32]
///
/// # 索引计算
///
/// 对于 (i, j) 其中 i > j:
/// - offset = i*(i-1)/2 + j
///
/// # 内存复杂度
///
/// - 距离数组: n*(n-1)/2 * 4 bytes
/// - 总计: O(n²) 空间
pub struct CompressedDistanceMatrix {
    /// 下三角距离数组（不包括对角线）
    ///
    /// 长度: n*(n-1)/2
    distances: Vec<Value>,

    /// 点的数量
    n: usize,
}

impl CompressedDistanceMatrix {
    /// 创建空的压缩距离矩阵
    ///
    /// # Arguments
    ///
    /// * `n` - 点的数量
    ///
    /// # Panics
    ///
    /// 当 n < 2 时 panic
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "Distance matrix requires at least 2 points");

        let size = n * (n - 1) / 2;
        Self {
            distances: vec![0.0; size],
            n,
        }
    }

    /// 从距离数组创建（一维数组，按行存储下三角）
    ///
    /// # Arguments
    ///
    /// * `distances` - 下三角距离数组（长度必须为 n*(n-1)/2）
    /// * `n` - 点的数量
    ///
    /// # Panics
    ///
    /// 当距离数组长度不匹配时 panic
    pub fn from_distances(distances: Vec<Value>, n: usize) -> Self {
        let expected_len = n * (n - 1) / 2;
        assert_eq!(
            distances.len(),
            expected_len,
            "Distance array length mismatch: expected {}, got {}",
            expected_len,
            distances.len()
        );

        Self { distances, n }
    }

    /// 从点云构建距离矩阵
    ///
    /// # Arguments
    ///
    /// * `points` - 点云数据（每个点是一个切片）
    /// * `metric` - 距离度量类型
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    /// let dist = CompressedDistanceMatrix::from_points(&points, Metric::Euclidean);
    /// ```
    pub fn from_points<P: AsRef<[f32]>>(points: &[P], metric: Metric) -> Self {
        let n = points.len();
        assert!(n >= 2, "Need at least 2 points");

        let size = n * (n - 1) / 2;
        let mut distances = vec![0.0; size];

        let mut idx = 0;
        for i in 1..n {
            for j in 0..i {
                let d = metric.distance(points[i].as_ref(), points[j].as_ref());
                distances[idx] = d;
                idx += 1;
            }
        }

        Self { distances, n }
    }

    /// 获取点的数量
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// 获取两点之间的距离
    ///
    /// # Arguments
    ///
    /// * `i` - 第一个点的索引
    /// * `j` - 第二个点的索引
    ///
    /// # Returns
    ///
    /// 距离值（如果 i == j 返回 0.0）
    ///
    /// # Panics
    ///
    /// 当索引越界时 panic
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Value {
        assert!(i < self.n, "Index i={} out of bounds (n={})", i, self.n);
        assert!(j < self.n, "Index j={} out of bounds (n={})", j, self.n);

        if i == j {
            return 0.0;
        }

        // 确保 i > j（下三角存储）
        let (row, col) = if i > j { (i, j) } else { (j, i) };

        let offset = row * (row - 1) / 2 + col;
        self.distances[offset]
    }

    /// 设置两点之间的距离
    ///
    /// # Arguments
    ///
    /// * `i` - 第一个点的索引
    /// * `j` - 第二个点的索引
    /// * `value` - 距离值
    ///
    /// # Panics
    ///
    /// 当索引越界或 i == j 时 panic
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: Value) {
        assert!(i < self.n, "Index i={} out of bounds (n={})", i, self.n);
        assert!(j < self.n, "Index j={} out of bounds (n={})", j, self.n);
        assert!(i != j, "Cannot set distance for i=j={}", i);

        // 确保 i > j（下三角存储）
        let (row, col) = if i > j { (i, j) } else { (j, i) };

        let offset = row * (row - 1) / 2 + col;
        self.distances[offset] = value;
    }

    /// 获取所有距离的最大值
    pub fn max_distance(&self) -> Value {
        self.distances
            .iter()
            .copied()
            .fold(0.0, f32::max)
    }

    /// 获取所有距离的最小值（不包括对角线的0）
    pub fn min_distance(&self) -> Value {
        self.distances
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
    }
}

impl fmt::Debug for CompressedDistanceMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CompressedDistanceMatrix {{")?;
        writeln!(f, "  n: {},", self.n)?;
        writeln!(f, "  distances: [")?;
        for i in 0..self.n.min(5) {
            write!(f, "    [")?;
            for j in 0..self.n.min(5) {
                write!(f, "{:6.3}", self.get(i, j))?;
                if j < self.n.min(5) - 1 {
                    write!(f, ", ")?;
                }
            }
            if self.n > 5 {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.n > 5 {
            writeln!(f, "    ...")?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "}}")
    }
}

// ============================================================================
// SparseDistanceMatrix - 稀疏矩阵（邻接表存储）
// ============================================================================

/// 稀疏距离矩阵（邻接表存储）
///
/// # 适用场景
///
/// - 大规模点云（n > 1000）
/// - 稀疏距离（使用阈值过滤）
/// - 内存受限环境
///
/// # 数据格式
///
/// 使用 COO (Coordinate) 格式输入：
/// ```text
/// edges = [(i, j, distance), ...]
/// ```
///
/// 内部转换为邻接表：
/// ```text
/// neighbors[i] = [(j, distance), ...]
/// ```
pub struct SparseDistanceMatrix {
    /// 邻接表：neighbors[i] = [(邻居索引, 距离), ...]
    neighbors: Vec<Vec<(usize, Value)>>,

    /// 点的数量
    n: usize,

    /// 最大距离（用于判断阈值）
    max_distance: Value,
}

impl SparseDistanceMatrix {
    /// 创建空的稀疏距离矩阵
    ///
    /// # Arguments
    ///
    /// * `n` - 点的数量
    pub fn new(n: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); n],
            n,
            max_distance: 0.0,
        }
    }

    /// 从 COO 格式边列表构建稀疏矩阵
    ///
    /// # Arguments
    ///
    /// * `edges` - 边列表 [(i, j, distance), ...]
    /// * `n` - 点的数量
    ///
    /// # 注意
    ///
    /// - 自动对称化（添加 (j, i) 如果只有 (i, j)）
    /// - 自动去重（保留较小的距离）
    pub fn from_edges(edges: &[(usize, usize, Value)], n: usize) -> Self {
        let mut matrix = Self::new(n);

        for &(i, j, d) in edges {
            assert!(i < n, "Index i={} out of bounds (n={})", i, n);
            assert!(j < n, "Index j={} out of bounds (n={})", j, n);
            assert!(i != j, "Self-loops not allowed: i=j={}", i);

            // 添加双向边
            matrix.add_edge(i, j, d);
            matrix.add_edge(j, i, d);

            // 更新最大距离
            if d > matrix.max_distance {
                matrix.max_distance = d;
            }
        }

        // 排序并去重每个邻接表
        for neighbors in &mut matrix.neighbors {
            neighbors.sort_by(|a, b| a.0.cmp(&b.0));
            neighbors.dedup_by(|a, b| {
                if a.0 == b.0 {
                    // 保留较小的距离
                    b.1 = a.1.min(b.1);
                    true
                } else {
                    false
                }
            });
        }

        matrix
    }

    /// 从点云构建稀疏矩阵（使用距离阈值）
    ///
    /// # Arguments
    ///
    /// * `points` - 点云数据
    /// * `metric` - 距离度量
    /// * `threshold` - 距离阈值（仅保留距离 ≤ threshold 的边）
    pub fn from_points_with_threshold<P: AsRef<[f32]>>(
        points: &[P],
        metric: Metric,
        threshold: Value,
    ) -> Self {
        let n = points.len();
        let mut matrix = Self::new(n);

        for i in 0..n {
            for j in 0..i {
                let d = metric.distance(points[i].as_ref(), points[j].as_ref());
                if d <= threshold {
                    matrix.add_edge(i, j, d);
                    matrix.add_edge(j, i, d);

                    if d > matrix.max_distance {
                        matrix.max_distance = d;
                    }
                }
            }
        }

        matrix
    }

    /// 添加一条边
    #[inline]
    fn add_edge(&mut self, i: usize, j: usize, d: Value) {
        self.neighbors[i].push((j, d));
    }

    /// 获取点的数量
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// 获取两点之间的距离
    ///
    /// # Returns
    ///
    /// - 如果边存在，返回距离
    /// - 如果 i == j，返回 0.0
    /// - 如果边不存在，返回 f32::INFINITY
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Value {
        if i == j {
            return 0.0;
        }

        for &(neighbor, dist) in &self.neighbors[i] {
            if neighbor == j {
                return dist;
            }
        }

        f32::INFINITY
    }

    /// 获取某个点的所有邻居
    #[inline]
    pub fn get_neighbors(&self, i: usize) -> &[(usize, Value)] {
        &self.neighbors[i]
    }

    /// 获取最大距离
    #[inline]
    pub fn get_max_distance(&self) -> Value {
        self.max_distance
    }

    /// 获取边的总数（无向图，每条边计数一次）
    pub fn num_edges(&self) -> usize {
        self.neighbors.iter().map(|n| n.len()).sum::<usize>() / 2
    }
}

impl fmt::Debug for SparseDistanceMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SparseDistanceMatrix {{")?;
        writeln!(f, "  n: {},", self.n)?;
        writeln!(f, "  edges: {},", self.num_edges())?;
        writeln!(f, "  max_distance: {:.3},", self.max_distance)?;
        writeln!(f, "  neighbors: [")?;
        for i in 0..self.n.min(5) {
            writeln!(f, "    {}: {:?}", i, self.neighbors[i])?;
        }
        if self.n > 5 {
            writeln!(f, "    ...")?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "}}")
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_euclidean() {
        let p1 = [0.0, 0.0];
        let p2 = [3.0, 4.0];
        let d = Metric::Euclidean.distance(&p1, &p2);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_manhattan() {
        let p1 = [0.0, 0.0];
        let p2 = [3.0, 4.0];
        let d = Metric::Manhattan.distance(&p1, &p2);
        assert!((d - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_chebyshev() {
        let p1 = [0.0, 0.0];
        let p2 = [3.0, 4.0];
        let d = Metric::Chebyshev.distance(&p1, &p2);
        assert!((d - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_matrix_basic() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let dist = CompressedDistanceMatrix::from_points(&points, Metric::Euclidean);

        assert_eq!(dist.size(), 4);

        // 点0和点1之间的距离应该是1.0
        assert!((dist.get(0, 1) - 1.0).abs() < 1e-6);

        // 对称性
        assert!((dist.get(0, 1) - dist.get(1, 0)).abs() < 1e-6);

        // 对角线应该是0
        assert!((dist.get(0, 0)).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_matrix_set_get() {
        let mut dist = CompressedDistanceMatrix::new(3);

        dist.set(1, 0, 1.5);
        dist.set(2, 0, 2.5);
        dist.set(2, 1, 3.5);

        assert!((dist.get(0, 1) - 1.5).abs() < 1e-6);
        assert!((dist.get(1, 0) - 1.5).abs() < 1e-6);
        assert!((dist.get(2, 1) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_matrix_max_min() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dist = CompressedDistanceMatrix::from_points(&points, Metric::Euclidean);

        let max_d = dist.max_distance();
        let min_d = dist.min_distance();

        assert!(max_d >= min_d);
        assert!(min_d > 0.0);
    }

    #[test]
    fn test_sparse_matrix_from_edges() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
        let dist = SparseDistanceMatrix::from_edges(&edges, 3);

        assert_eq!(dist.size(), 3);
        assert_eq!(dist.num_edges(), 3);

        assert!((dist.get(0, 1) - 1.0).abs() < 1e-6);
        assert!((dist.get(1, 2) - 2.0).abs() < 1e-6);
        assert!((dist.get(0, 2) - 3.0).abs() < 1e-6);

        // 对称性
        assert!((dist.get(0, 1) - dist.get(1, 0)).abs() < 1e-6);

        // 不存在的边
        assert_eq!(dist.get(0, 3), f32::INFINITY);
    }

    #[test]
    fn test_sparse_matrix_threshold() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let dist = SparseDistanceMatrix::from_points_with_threshold(
            &points,
            Metric::Euclidean,
            2.0,
        );

        // 点3距离其他点很远，不应该有边
        assert_eq!(dist.get_neighbors(3).len(), 0);

        // 点0、1、2之间应该有边
        assert!(dist.get_neighbors(0).len() > 0);
    }

    #[test]
    fn test_sparse_matrix_neighbors() {
        let edges = vec![(0, 1, 1.0), (0, 2, 2.0)];
        let dist = SparseDistanceMatrix::from_edges(&edges, 3);

        let neighbors = dist.get_neighbors(0);
        assert_eq!(neighbors.len(), 2);

        // 应该已排序
        assert!(neighbors[0].0 < neighbors[1].0);
    }

    #[test]
    #[should_panic(expected = "at least 2 points")]
    fn test_compressed_matrix_too_few_points() {
        CompressedDistanceMatrix::new(1);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_compressed_matrix_out_of_bounds() {
        let dist = CompressedDistanceMatrix::new(3);
        dist.get(3, 0);
    }

    #[test]
    #[should_panic(expected = "Cannot set distance for i=j")]
    fn test_compressed_matrix_set_diagonal() {
        let mut dist = CompressedDistanceMatrix::new(3);
        dist.set(1, 1, 1.0);
    }
}
