/*!
 * Simplex 编解码与枚举器
 *
 * 这个模块实现了简单复形的组合编码/解码算法以及边界/余边界枚举器。
 *
 * # 核心概念
 *
 * ## 组合编码
 *
 * k-simplex 由 k+1 个顶点组成，按升序排列。我们使用二项式系数将其编码为一个整数。
 *
 * 对于 k-simplex {v0, v1, ..., vk}（v0 < v1 < ... < vk），其索引为：
 * ```text
 * index = C(vk, k+1) + C(v_{k-1}, k) + ... + C(v1, 2) + v0
 * ```
 *
 * ## 示例
 *
 * - 边 {0, 1}: index = C(1, 2) + 0 = 0
 * - 边 {0, 2}: index = C(2, 2) + 0 = 1
 * - 边 {1, 2}: index = C(2, 2) + 1 = 2
 * - 三角形 {0, 1, 2}: index = C(2, 3) + C(1, 2) + 0 = 0
 *
 * # 使用示例
 *
 * ```ignore
 * use rust_indicators::ripser::core::{BinomialCoeffTable, get_edge_index, get_edge_vertices};
 *
 * let binomial = BinomialCoeffTable::new(100, 50);
 *
 * // 编码边 {3, 5}
 * let index = get_edge_index(5, 3, &binomial);
 *
 * // 解码边
 * let (i, j) = get_edge_vertices(index, &binomial);
 * assert_eq!((i, j), (5, 3));
 * ```
 */

use super::super::types::{DiameterEntry, Index, Value};
use super::binomial::BinomialCoeffTable;
use super::distance::{CompressedDistanceMatrix, SparseDistanceMatrix};
use std::cmp::Ordering;

// ============================================================================
// Simplex 编码/解码
// ============================================================================

/// 获取边的组合编码索引
///
/// 对于边 {i, j}（i > j），其索引为 C(i, 2) + j
///
/// # Arguments
///
/// * `i` - 较大的顶点索引
/// * `j` - 较小的顶点索引
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// 边的组合编码索引
///
/// # Panics
///
/// 当 j >= i 时 panic
#[inline]
pub fn get_edge_index(i: usize, j: usize, binomial: &BinomialCoeffTable) -> Index {
    assert!(i > j, "Invalid edge: i={} must be > j={}", i, j);
    binomial.get(i, 2) + j as Index
}

/// 获取边的顶点
///
/// 解码边索引，返回 (i, j) 其中 i > j
///
/// # Arguments
///
/// * `index` - 边的组合编码索引
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// (i, j) 元组，其中 i > j
#[inline]
pub fn get_edge_vertices(index: Index, binomial: &BinomialCoeffTable) -> (usize, usize) {
    // 对于 k=2，可以使用精确公式
    // C(i, 2) = i*(i-1)/2 <= index
    // 求解: i = ceil((1 + sqrt(1 + 8*index)) / 2)
    let i = get_max_vertex_for_edge(index);
    let j = (index - binomial.get(i, 2)) as usize;
    (i, j)
}

/// 使用精确公式获取边的最大顶点
///
/// 求解 C(i, 2) <= index，返回最大的 i
///
/// 使用公式: i = floor((1 + sqrt(1 + 8*index)) / 2)
#[inline]
fn get_max_vertex_for_edge(index: Index) -> usize {
    // C(i, 2) = i*(i-1)/2
    // 求解: i^2 - i - 2*index = 0
    // i = (1 + sqrt(1 + 8*index)) / 2
    let sqrt_arg = 1.0 + 8.0 * index as f64;
    let i = ((1.0 + sqrt_arg.sqrt()) / 2.0).floor() as usize;
    i
}

/// 通用 simplex 解码
///
/// 将索引解码为顶点列表
///
/// # Arguments
///
/// * `index` - simplex 的组合编码索引
/// * `dim` - simplex 的维度（k-simplex 有 k+1 个顶点）
/// * `n` - 最大顶点索引 + 1
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// 顶点列表（升序）
///
/// # 示例
///
/// ```ignore
/// // 解码 2-simplex（三角形）
/// let vertices = get_simplex_vertices(index, 2, 10, &binomial);
/// assert_eq!(vertices.len(), 3);  // 3 个顶点
/// ```
pub fn get_simplex_vertices(
    mut index: Index,
    dim: usize,
    mut n: usize,
    binomial: &BinomialCoeffTable,
) -> Vec<usize> {
    let mut vertices = vec![0; dim + 1];
    n = n.saturating_sub(1);

    // 从最高维度开始解码
    for k in (1..=dim).rev() {
        // 找最大的 v 使得 C(v, k+1) <= index
        let v = get_max_vertex(index, k + 1, n, binomial);
        vertices[k] = v;
        index -= binomial.get(v, k + 1);
        n = v;
    }

    vertices[0] = index as usize;
    vertices
}

/// 获取最大顶点（通用版本）
///
/// 找最大的 v 使得 C(v, k) <= index
///
/// # Arguments
///
/// * `index` - 目标索引
/// * `k` - 二项式系数的 k
/// * `n` - 搜索上界
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// 最大的 v 使得 C(v, k) <= index
fn get_max_vertex(index: Index, k: usize, n: usize, binomial: &BinomialCoeffTable) -> usize {
    // 特殊情况：k=2 使用精确公式
    if k == 2 {
        return get_max_vertex_for_edge(index);
    }

    // 通用情况：二分查找
    let mut low = 0;
    let mut high = n;

    while low < high {
        let mid = (low + high + 1) / 2;
        if binomial.get(mid, k) <= index {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

/// 通用 simplex 编码
///
/// 将顶点列表编码为索引
///
/// 编码公式: index = C(v_k, k+1) + C(v_{k-1}, k) + ... + C(v_1, 2) + v_0
/// 其中 v_0 < v_1 < ... < v_k
///
/// # Arguments
///
/// * `vertices` - 顶点列表（必须升序）
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// simplex 的组合编码索引
///
/// # Panics
///
/// 当顶点列表未排序时 panic
pub fn encode_simplex(vertices: &[usize], binomial: &BinomialCoeffTable) -> Index {
    // 验证升序
    for i in 1..vertices.len() {
        assert!(
            vertices[i] > vertices[i - 1],
            "Vertices must be in ascending order: {:?}",
            vertices
        );
    }

    let mut index: Index = 0;

    // 编码公式: C(v_k, k+1) + C(v_{k-1}, k) + ... + C(v_1, 2) + v_0
    for (k, &v) in vertices.iter().enumerate().skip(1) {
        index += binomial.get(v, k + 1);
    }
    index += vertices[0] as Index;

    index
}

// ============================================================================
// 边枚举器
// ============================================================================

/// 边枚举器（从压缩距离矩阵）
///
/// 枚举所有距离 ≤ threshold 的边，按直径升序排序
pub struct EdgeEnumerator {
    edges: Vec<DiameterEntry>,
    index: usize,
}

impl EdgeEnumerator {
    /// 从压缩距离矩阵创建边枚举器
    ///
    /// # Arguments
    ///
    /// * `dist` - 压缩距离矩阵
    /// * `threshold` - 距离阈值
    /// * `binomial` - 二项式系数表
    pub fn from_compressed_matrix(
        dist: &CompressedDistanceMatrix,
        threshold: Value,
        binomial: &BinomialCoeffTable,
    ) -> Self {
        let n = dist.size();
        let mut edges = Vec::new();

        for i in 1..n {
            for j in 0..i {
                let d = dist.get(i, j);
                if d <= threshold {
                    let index = get_edge_index(i, j, binomial);
                    edges.push(DiameterEntry::new(d, index, 0));
                }
            }
        }

        // 按直径排序
        edges.sort_by(|a, b| {
            a.diameter
                .partial_cmp(&b.diameter)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });

        Self { edges, index: 0 }
    }

    /// 从稀疏距离矩阵创建边枚举器
    ///
    /// # Arguments
    ///
    /// * `dist` - 稀疏距离矩阵
    /// * `binomial` - 二项式系数表
    pub fn from_sparse_matrix(
        dist: &SparseDistanceMatrix,
        binomial: &BinomialCoeffTable,
    ) -> Self {
        let n = dist.size();
        let mut edges = Vec::new();

        for i in 0..n {
            for &(j, d) in dist.get_neighbors(i) {
                if i > j {
                    // 避免重复（只处理 i > j）
                    let index = get_edge_index(i, j, binomial);
                    edges.push(DiameterEntry::new(d, index, 0));
                }
            }
        }

        // 按直径排序
        edges.sort_by(|a, b| {
            a.diameter
                .partial_cmp(&b.diameter)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });

        Self { edges, index: 0 }
    }

    /// 获取所有边（已排序）
    pub fn get_edges(&self) -> &[DiameterEntry] {
        &self.edges
    }

    /// 获取边的数量
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// 判断是否为空
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

impl Iterator for EdgeEnumerator {
    type Item = DiameterEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.edges.len() {
            let edge = self.edges[self.index];
            self.index += 1;
            Some(edge)
        } else {
            None
        }
    }
}

// ============================================================================
// 边界枚举器
// ============================================================================

/// Simplex 边界枚举器
///
/// 枚举 k-simplex 的所有 (k-1)-face（边界）
///
/// # 示例
///
/// 对于 2-simplex {0, 1, 2}（三角形），其边界为：
/// - {1, 2}
/// - {0, 2}
/// - {0, 1}
pub struct SimplexBoundaryEnumerator {
    /// 当前 simplex 的索引
    #[allow(dead_code)]
    simplex_index: Index,
    /// simplex 的维度
    dim: usize,
    /// 当前枚举位置（使用 isize 避免下溢）
    k: isize,
    /// 索引下界（用于解码）
    idx_below: Index,
    /// 索引上界（用于解码）
    idx_above: Index,
    /// 当前顶点索引
    j: usize,
}

impl SimplexBoundaryEnumerator {
    /// 创建边界枚举器
    ///
    /// # Arguments
    ///
    /// * `simplex_index` - simplex 的组合编码索引
    /// * `dim` - simplex 的维度
    /// * `n` - 最大顶点索引 + 1
    pub fn new(simplex_index: Index, dim: usize, n: usize) -> Self {
        Self {
            simplex_index,
            dim,
            k: dim as isize,
            idx_below: simplex_index,
            idx_above: 0,
            j: n - 1,
        }
    }

    /// 获取下一个边界 face
    ///
    /// # Arguments
    ///
    /// * `binomial` - 二项式系数表
    ///
    /// # Returns
    ///
    /// 边界 face 的 DiameterEntry（diameter 暂时为 0，需要后续计算）
    pub fn next_boundary(&mut self, binomial: &BinomialCoeffTable) -> Option<DiameterEntry> {
        // k 从 dim 递减到 0，共 dim+1 次迭代
        if self.k < 0 {
            return None;
        }

        let k_usize = self.k as usize;

        // 解码找到下一个顶点 j
        self.j = get_max_vertex(self.idx_below, k_usize + 1, self.j, binomial);

        // 计算 face 的索引
        let face_index =
            self.idx_above - binomial.get(self.j, k_usize + 1) + self.idx_below;

        // 计算符号（交替，Z/2Z 中忽略）
        let coefficient = 1; // 在 Z/2Z 中系数总是 1

        // 更新状态
        self.idx_below -= binomial.get(self.j, k_usize + 1);
        self.idx_above += binomial.get(self.j, k_usize);
        self.k -= 1;

        Some(DiameterEntry::new(0.0, face_index, coefficient))
    }

    /// 获取所有边界 face
    pub fn get_all_boundaries(
        &mut self,
        binomial: &BinomialCoeffTable,
    ) -> Vec<DiameterEntry> {
        let mut boundaries = Vec::with_capacity(self.dim + 1);
        while let Some(face) = self.next_boundary(binomial) {
            boundaries.push(face);
        }
        boundaries
    }
}

// ============================================================================
// 余边界枚举器
// ============================================================================

/// Simplex 余边界枚举器
///
/// 枚举所有包含给定 k-simplex 的 (k+1)-simplex（余边界）
///
/// # 示例
///
/// 对于边 {0, 1}，其余边界包括所有包含它的三角形：
/// - {0, 1, 2}
/// - {0, 1, 3}
/// - ...
pub struct SimplexCoboundaryEnumerator {
    /// simplex 的顶点
    vertices: Vec<usize>,
    /// simplex 的维度
    #[allow(dead_code)]
    dim: usize,
    /// 下一个候选顶点
    next_vertex: usize,
    /// 最大顶点索引
    max_vertex: usize,
}

impl SimplexCoboundaryEnumerator {
    /// 创建余边界枚举器
    ///
    /// # Arguments
    ///
    /// * `simplex_index` - simplex 的组合编码索引
    /// * `dim` - simplex 的维度
    /// * `n` - 最大顶点索引 + 1
    /// * `binomial` - 二项式系数表
    pub fn new(
        simplex_index: Index,
        dim: usize,
        n: usize,
        binomial: &BinomialCoeffTable,
    ) -> Self {
        let vertices = get_simplex_vertices(simplex_index, dim, n, binomial);
        let next_vertex = vertices[dim] + 1; // 从最大顶点的下一个开始

        Self {
            vertices,
            dim,
            next_vertex,
            max_vertex: n,
        }
    }

    /// 获取下一个余边界 cofacet
    ///
    /// # Arguments
    ///
    /// * `dist` - 距离矩阵（用于计算直径）
    /// * `binomial` - 二项式系数表
    ///
    /// # Returns
    ///
    /// 余边界 cofacet 的 DiameterEntry
    pub fn next_coboundary(
        &mut self,
        dist: &CompressedDistanceMatrix,
        binomial: &BinomialCoeffTable,
    ) -> Option<DiameterEntry> {
        if self.next_vertex >= self.max_vertex {
            return None;
        }

        let v = self.next_vertex;
        self.next_vertex += 1;

        // 构建 cofacet 的顶点列表
        let mut cofacet_vertices = self.vertices.clone();
        cofacet_vertices.push(v);
        cofacet_vertices.sort_unstable();

        // 计算直径（所有边的最大距离）
        let diameter = compute_diameter(&cofacet_vertices, dist);

        // 编码为索引
        let index = encode_simplex(&cofacet_vertices, binomial);

        Some(DiameterEntry::new(diameter, index, 1))
    }

    /// 获取所有余边界 cofacet（距离 ≤ threshold）
    pub fn get_all_coboundaries(
        &mut self,
        dist: &CompressedDistanceMatrix,
        binomial: &BinomialCoeffTable,
        threshold: Value,
    ) -> Vec<DiameterEntry> {
        let mut coboundaries = Vec::new();
        while let Some(cofacet) = self.next_coboundary(dist, binomial) {
            if cofacet.diameter <= threshold {
                coboundaries.push(cofacet);
            }
        }
        coboundaries
    }
}

/// 计算 simplex 的直径
///
/// 直径定义为所有边的最大距离
fn compute_diameter(vertices: &[usize], dist: &CompressedDistanceMatrix) -> Value {
    let mut max_dist = 0.0;
    for i in 0..vertices.len() {
        for j in 0..i {
            let d = dist.get(vertices[i], vertices[j]);
            if d > max_dist {
                max_dist = d;
            }
        }
    }
    max_dist
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_binomial() -> BinomialCoeffTable {
        // 使用安全的最大值（避免溢出）
        // C(57, 28) < 2^55
        BinomialCoeffTable::new(57, 28)
    }

    #[test]
    fn test_edge_encode_decode() {
        let binomial = get_test_binomial();

        // 测试几条边
        let edges = vec![(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];

        for (i, j) in edges {
            let index = get_edge_index(i, j, &binomial);
            let (i2, j2) = get_edge_vertices(index, &binomial);
            assert_eq!(
                (i, j),
                (i2, j2),
                "Edge ({}, {}) encode-decode failed: index={}, decoded=({}, {})",
                i,
                j,
                index,
                i2,
                j2
            );
        }
    }

    #[test]
    fn test_edge_index_formula() {
        let binomial = get_test_binomial();

        // 验证索引公式: C(i, 2) + j
        for i in 1..10 {
            for j in 0..i {
                let index = get_edge_index(i, j, &binomial);
                let expected = binomial.get(i, 2) + j as Index;
                assert_eq!(
                    index, expected,
                    "Edge ({}, {}) index formula failed",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_simplex_encode_decode() {
        let binomial = get_test_binomial();

        // 测试三角形
        let triangle = vec![0, 1, 2];
        let index = encode_simplex(&triangle, &binomial);
        let decoded = get_simplex_vertices(index, 2, 10, &binomial);
        assert_eq!(triangle, decoded, "Triangle encode-decode failed");

        // 测试四面体
        let tetrahedron = vec![0, 1, 2, 3];
        let index = encode_simplex(&tetrahedron, &binomial);
        let decoded = get_simplex_vertices(index, 3, 10, &binomial);
        assert_eq!(tetrahedron, decoded, "Tetrahedron encode-decode failed");
    }

    #[test]
    fn test_edge_enumerator_compressed() {
        let binomial = get_test_binomial();

        // 创建一个简单的距离矩阵
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let dist = CompressedDistanceMatrix::from_points(
            &points,
            super::super::distance::Metric::Euclidean,
        );

        // 枚举所有边（threshold = 2.0）
        let enumerator = EdgeEnumerator::from_compressed_matrix(&dist, 2.0, &binomial);

        // 应该有 C(4, 2) = 6 条边
        assert_eq!(enumerator.len(), 6, "Should have 6 edges");

        // 验证排序（按直径升序）
        let edges = enumerator.get_edges();
        for i in 1..edges.len() {
            assert!(
                edges[i - 1].diameter <= edges[i].diameter,
                "Edges not sorted by diameter"
            );
        }
    }

    #[test]
    fn test_boundary_enumerator() {
        let binomial = get_test_binomial();

        // 三角形 {0, 1, 2}
        let triangle = vec![0, 1, 2];
        let index = encode_simplex(&triangle, &binomial);

        let mut enumerator = SimplexBoundaryEnumerator::new(index, 2, 10);
        let boundaries = enumerator.get_all_boundaries(&binomial);

        // 三角形应该有 3 条边
        assert_eq!(boundaries.len(), 3, "Triangle should have 3 edges");

        // 解码每条边并验证
        let mut edge_vertices: Vec<_> = boundaries
            .iter()
            .map(|e| get_edge_vertices(e.index, &binomial))
            .collect();
        edge_vertices.sort();

        let expected = vec![(1, 0), (2, 0), (2, 1)];
        assert_eq!(edge_vertices, expected, "Boundary edges incorrect");
    }

    #[test]
    fn test_coboundary_enumerator() {
        let binomial = get_test_binomial();

        // 创建距离矩阵
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let dist = CompressedDistanceMatrix::from_points(
            &points,
            super::super::distance::Metric::Euclidean,
        );

        // 边 {0, 1}
        let edge_index = get_edge_index(1, 0, &binomial);

        let mut enumerator =
            SimplexCoboundaryEnumerator::new(edge_index, 1, 4, &binomial);
        let coboundaries =
            enumerator.get_all_coboundaries(&dist, &binomial, f32::INFINITY);

        // 边 {0, 1} 应该包含在 2 个三角形中：{0, 1, 2} 和 {0, 1, 3}
        assert_eq!(coboundaries.len(), 2, "Edge {{0,1}} should be in 2 triangles");
    }

    #[test]
    fn test_compute_diameter() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dist = CompressedDistanceMatrix::from_points(
            &points,
            super::super::distance::Metric::Euclidean,
        );

        // 三角形 {0, 1, 2}
        let vertices = vec![0, 1, 2];
        let diameter = compute_diameter(&vertices, &dist);

        // 直径应该是最大边长（对角线）
        let max_edge = dist.get(1, 2); // 边 {1, 2} 的长度
        assert!(
            (diameter - max_edge).abs() < 1e-6 || diameter >= max_edge,
            "Diameter should be max edge length"
        );
    }

    #[test]
    #[should_panic(expected = "must be >")]
    fn test_edge_index_invalid() {
        let binomial = get_test_binomial();
        get_edge_index(1, 2, &binomial); // j >= i, should panic
    }

    #[test]
    #[should_panic(expected = "ascending order")]
    fn test_encode_simplex_unsorted() {
        let binomial = get_test_binomial();
        let vertices = vec![2, 1, 0]; // 未排序
        encode_simplex(&vertices, &binomial);
    }

    #[test]
    fn test_edge_vertices_all_pairs() {
        let binomial = get_test_binomial();

        // 验证所有 n=5 的边
        let n = 5;
        let mut index = 0;

        for i in 1..n {
            for j in 0..i {
                let calculated_index = get_edge_index(i, j, &binomial);
                assert_eq!(
                    calculated_index, index,
                    "Edge ({}, {}) index should be {}",
                    i, j, index
                );

                let (i2, j2) = get_edge_vertices(index, &binomial);
                assert_eq!((i, j), (i2, j2), "Decode edge {} failed", index);

                index += 1;
            }
        }
    }

    #[test]
    fn test_simplex_consistency() {
        let binomial = get_test_binomial();

        // 测试多个不同维度的 simplex
        let test_cases = vec![
            vec![0, 1],          // 边
            vec![0, 1, 2],       // 三角形
            vec![0, 1, 2, 3],    // 四面体
            vec![1, 3, 5, 7, 9], // 5-simplex
        ];

        for vertices in test_cases {
            let dim = vertices.len() - 1;
            let index = encode_simplex(&vertices, &binomial);
            let decoded = get_simplex_vertices(index, dim, 20, &binomial);
            assert_eq!(
                vertices, decoded,
                "Simplex {:?} encode-decode failed",
                vertices
            );
        }
    }
}
