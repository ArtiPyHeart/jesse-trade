/*!
 * Ripser 算法 - 端到端实现
 *
 * 这个模块实现完整的 Ripser 算法，整合所有核心组件：
 * - 距离矩阵计算
 * - 简单复形枚举
 * - 持久性同调计算
 *
 * # 算法流程
 *
 * 1. 计算距离矩阵（或接受稀疏距离矩阵）
 * 2. 枚举简单复形（边、三角形等）
 * 3. 计算 0 维同调（连通分量）
 * 4. 计算高维同调（环、空洞等）
 * 5. 返回持久性条形码
 *
 * # 使用示例
 *
 * ```ignore
 * use ripser::compute_ripser;
 *
 * let points = vec![
 *     vec![0.0, 0.0],
 *     vec![1.0, 0.0],
 *     vec![0.0, 1.0],
 * ];
 *
 * let result = compute_ripser(&points, 2, None, "euclidean");
 * ```
 */

use super::binomial::BinomialCoeffTable;
use super::cohomology::compute_dim_0_pairs;
use super::distance::{CompressedDistanceMatrix, Metric};
use super::simplex::{EdgeEnumerator, encode_simplex};
use super::super::types::{DiameterEntry, Index, Value};

// ============================================================================
// Ripser 结果
// ============================================================================

/// Ripser 计算结果
#[derive(Debug, Clone)]
pub struct RipserResult {
    /// 各维度的持久性对
    /// dimension -> [(birth, death), ...]
    pub persistence: Vec<Vec<(Value, Value)>>,

    /// 使用的距离矩阵大小
    pub num_points: usize,

    /// 计算的最大维度
    pub max_dim: usize,

    /// 使用的阈值
    pub threshold: Value,
}

impl RipserResult {
    /// 创建新的结果
    pub fn new(num_points: usize, max_dim: usize, threshold: Value) -> Self {
        Self {
            persistence: vec![Vec::new(); max_dim + 1],
            num_points,
            max_dim,
            threshold,
        }
    }

    /// 获取指定维度的持久性对
    pub fn get_persistence(&self, dim: usize) -> Option<&Vec<(Value, Value)>> {
        self.persistence.get(dim)
    }

    /// 设置指定维度的持久性对
    pub fn set_persistence(&mut self, dim: usize, pairs: Vec<(Value, Value)>) {
        if dim < self.persistence.len() {
            self.persistence[dim] = pairs;
        }
    }

    /// 获取所有持久性对的总数
    pub fn total_pairs(&self) -> usize {
        self.persistence.iter().map(|v| v.len()).sum()
    }
}

// ============================================================================
// 距离矩阵输入
// ============================================================================

/// 距离矩阵输入类型
pub enum DistanceInput<'a> {
    /// 点云数据（每行是一个点）
    Points(&'a [Vec<f64>]),

    /// 压缩距离矩阵（下三角）
    Compressed(&'a [f64]),
}

// ============================================================================
// Simplex Filtration
// ============================================================================

/// Simplex Filtration 项
///
/// 表示 filtration 中的一个 simplex，包含所有必要信息
#[derive(Debug, Clone)]
struct FiltrationSimplex {
    /// Simplex 的顶点（降序排列）
    vertices: Vec<usize>,

    /// Simplex 的直径（filtration 值）
    diameter: Value,

    /// Simplex 的组合编码索引
    combinatorial_index: Index,

    /// 全局 filtration 索引（连续分配）
    global_index: usize,
}

/// Simplex Filtration 构建器
///
/// 构建完整的 Vietoris-Rips filtration，包括所有维度的 simplices
struct SimplexFiltration {
    /// 所有 simplices（按直径排序）
    simplices: Vec<FiltrationSimplex>,

    /// 每个维度的起始索引
    /// dim_offsets[d] = 第 d 维的第一个 simplex 在 simplices 中的索引
    /// 注：Bug 3 修复后暂未使用（改用 filter-based），保留以备将来缓存优化使用
    #[allow(dead_code)]
    dim_offsets: Vec<usize>,
}

impl SimplexFiltration {
    /// 构建 Simplex Filtration
    ///
    /// # Arguments
    ///
    /// * `distance_matrix` - 距离矩阵
    /// * `max_dim` - 最大维度
    /// * `threshold` - 距离阈值
    /// * `binomial` - 二项式系数表
    fn build(
        distance_matrix: &CompressedDistanceMatrix,
        max_dim: usize,
        threshold: Value,
        binomial: &BinomialCoeffTable,
    ) -> Self {
        let n = distance_matrix.size();
        let mut all_simplices = Vec::new();
        let mut dim_offsets = vec![0];

        // 1. 添加顶点（0-simplices）
        for i in 0..n {
            all_simplices.push(FiltrationSimplex {
                vertices: vec![i],
                diameter: 0.0,
                combinatorial_index: i as Index,
                global_index: i,
            });
        }
        dim_offsets.push(all_simplices.len());

        // 2. 添加边（1-simplices）
        let edge_enumerator = EdgeEnumerator::from_compressed_matrix(
            distance_matrix,
            threshold,
            binomial,
        );
        let edges = edge_enumerator.get_edges();

        for edge in edges.iter() {
            let (i, j) = super::simplex::get_edge_vertices(edge.index, binomial);
            all_simplices.push(FiltrationSimplex {
                vertices: vec![i, j],
                diameter: edge.diameter,
                combinatorial_index: edge.index,
                global_index: all_simplices.len(),
            });
        }
        dim_offsets.push(all_simplices.len());

        // 3. 添加三角形（2-simplices）
        // 注意：计算 H_d 需要 (d+1)-simplices，所以 max_dim >= 1 时需要三角形
        let mut triangle_list = Vec::new();
        if max_dim >= 1 {
            // 枚举三角形
            let triangles = Self::enumerate_triangles(
                distance_matrix,
                &edges,
                threshold,
                binomial,
            );

            for triangle in triangles {
                triangle_list.push(triangle.clone());  // 保存用于四面体枚举
                all_simplices.push(FiltrationSimplex {
                    vertices: triangle.vertices,
                    diameter: triangle.diameter,
                    combinatorial_index: triangle.combinatorial_index,
                    global_index: all_simplices.len(),
                });
            }
            dim_offsets.push(all_simplices.len());
        }

        // 4. 添加四面体（3-simplices）
        // 计算 H_2 需要 3-simplices
        if max_dim >= 2 {
            // 枚举四面体
            let tetrahedra = Self::enumerate_tetrahedra(
                distance_matrix,
                &triangle_list,
                threshold,
                binomial,
            );

            for tetrahedron in tetrahedra {
                all_simplices.push(FiltrationSimplex {
                    vertices: tetrahedron.vertices,
                    diameter: tetrahedron.diameter,
                    combinatorial_index: tetrahedron.combinatorial_index,
                    global_index: all_simplices.len(),
                });
            }
            dim_offsets.push(all_simplices.len());
        }

        // 4. 按 (直径, 维度) 排序
        // 关键：相同直径时，低维 simplex 必须在高维之前（face before coface）
        all_simplices.sort_by(|a, b| {
            a.diameter
                .partial_cmp(&b.diameter)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.vertices.len().cmp(&b.vertices.len()))  // 维度升序
                .then_with(|| a.global_index.cmp(&b.global_index))      // 稳定排序
        });

        // 5. 重新分配全局索引
        for (idx, simplex) in all_simplices.iter_mut().enumerate() {
            simplex.global_index = idx;
        }

        // 6. 重新计算 dim_offsets（排序后维度的起始位置）
        dim_offsets.clear();
        dim_offsets.push(0);
        let mut current_dim = 0;
        for (idx, simplex) in all_simplices.iter().enumerate() {
            let simplex_dim = simplex.vertices.len() - 1;
            while current_dim < simplex_dim {
                dim_offsets.push(idx);
                current_dim += 1;
            }
        }
        // 添加最后的边界
        dim_offsets.push(all_simplices.len());

        SimplexFiltration {
            simplices: all_simplices,
            dim_offsets,
        }
    }

    /// 枚举三角形
    fn enumerate_triangles(
        distance_matrix: &CompressedDistanceMatrix,
        edges: &[DiameterEntry],
        threshold: Value,
        binomial: &BinomialCoeffTable,
    ) -> Vec<FiltrationSimplex> {
        let mut triangles = Vec::new();
        let n = distance_matrix.size();

        // 对每条边，找第三个顶点形成三角形
        for edge in edges.iter() {
            let (i, j) = super::simplex::get_edge_vertices(edge.index, binomial);
            // 注意：get_edge_vertices 返回 i > j

            for k in 0..n {
                if k == i || k == j {
                    continue;
                }

                // 只考虑 k > i 的情况，避免重复
                // 因为 i > j，所以 k > i > j
                if k <= i {
                    continue;
                }

                // 计算三角形的直径（三条边的最大值）
                let d_ij = distance_matrix.get(i, j);
                let d_ik = distance_matrix.get(i, k);
                let d_jk = distance_matrix.get(j, k);

                let triangle_diameter = d_ij.max(d_ik).max(d_jk);

                // 检查阈值
                if triangle_diameter > threshold {
                    continue;
                }

                // 编码三角形（必须升序）
                // 由于 k > i > j，升序为 [j, i, k]
                let vertices = vec![j, i, k];
                let combinatorial_index = encode_simplex(&vertices, binomial);

                triangles.push(FiltrationSimplex {
                    vertices,
                    diameter: triangle_diameter,
                    combinatorial_index,
                    global_index: 0,  // 临时值，后续会重新分配
                });
            }
        }

        triangles
    }

    /// 枚举四面体（3-simplices）
    fn enumerate_tetrahedra(
        distance_matrix: &CompressedDistanceMatrix,
        triangles: &[FiltrationSimplex],
        threshold: Value,
        binomial: &BinomialCoeffTable,
    ) -> Vec<FiltrationSimplex> {
        let mut tetrahedra = Vec::new();
        let n = distance_matrix.size();

        // 对每个三角形，找第四个顶点形成四面体
        for triangle in triangles.iter() {
            // 三角形顶点已经是升序：j < i < k
            let j = triangle.vertices[0];
            let i = triangle.vertices[1];
            let k = triangle.vertices[2];

            for l in 0..n {
                if l == j || l == i || l == k {
                    continue;
                }

                // 只考虑 l > k 的情况，避免重复
                // 最终顺序：j < i < k < l
                if l <= k {
                    continue;
                }

                // 计算四面体的直径（6条边的最大值）
                // 已有的3条边：(j,i), (j,k), (i,k)
                // 新增的3条边：(j,l), (i,l), (k,l)
                let d_jl = distance_matrix.get(j, l);
                let d_il = distance_matrix.get(i, l);
                let d_kl = distance_matrix.get(k, l);

                // 四面体直径 = max of all 6 edges
                let tetrahedron_diameter = triangle.diameter  // max of (j,i), (j,k), (i,k)
                    .max(d_jl)
                    .max(d_il)
                    .max(d_kl);

                // 检查阈值
                if tetrahedron_diameter > threshold {
                    continue;
                }

                // 编码四面体（必须升序）
                // 由于 l > k > i > j，升序为 [j, i, k, l]
                let vertices = vec![j, i, k, l];
                let combinatorial_index = encode_simplex(&vertices, binomial);

                tetrahedra.push(FiltrationSimplex {
                    vertices,
                    diameter: tetrahedron_diameter,
                    combinatorial_index,
                    global_index: 0,  // 临时值，后续会重新分配
                });
            }
        }

        tetrahedra
    }

    /// 获取指定维度的 simplices
    ///
    /// 注意：由于 simplices 按 (diameter, dimension) 排序，
    /// 维度可能交错出现，因此不能简单用 dim_offsets 切片。
    /// 必须过滤整个数组。
    fn get_simplices_by_dim(&self, dim: usize) -> Vec<&FiltrationSimplex> {
        self.simplices
            .iter()
            .filter(|s| s.vertices.len() - 1 == dim)
            .collect()
    }

    /// 获取直径数组（用于矩阵归约）
    fn get_diameters(&self) -> Vec<Value> {
        self.simplices.iter().map(|s| s.diameter).collect()
    }
}

// ============================================================================
// Ripser 主算法
// ============================================================================

/// 计算持久性同调（完整流程）
///
/// # Arguments
///
/// * `distance_input` - 距离矩阵输入（点云或距离矩阵）
/// * `max_dim` - 计算的最大维度（0, 1, 2, ...）
/// * `threshold` - 距离阈值（None 表示无限制）
/// * `metric` - 距离度量（"euclidean", "manhattan", "chebyshev"）
///
/// # Returns
///
/// RipserResult 包含各维度的持久性对
///
/// # Example
///
/// ```ignore
/// let points = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
/// ];
///
/// let result = compute_ripser(
///     DistanceInput::Points(&points),
///     2,
///     Some(2.0),
///     "euclidean"
/// );
/// ```
pub fn compute_ripser(
    distance_input: DistanceInput,
    max_dim: usize,
    threshold: Option<Value>,
    metric: &str,
) -> Result<RipserResult, String> {
    // 1. 构建距离矩阵
    let (distance_matrix, num_points) = match distance_input {
        DistanceInput::Points(points) => {
            if points.is_empty() {
                return Err("Empty point cloud".to_string());
            }

            let n = points.len();
            let metric_fn = parse_metric(metric)?;

            // 转换为 f32 向量
            let points_f32: Vec<Vec<f32>> = points
                .iter()
                .map(|p| p.iter().map(|&x| x as f32).collect())
                .collect();

            let dm = CompressedDistanceMatrix::from_points(&points_f32, metric_fn);
            (dm, n)
        }
        DistanceInput::Compressed(distances) => {
            // 从压缩距离推导点数
            // n*(n-1)/2 = distances.len()
            // n^2 - n - 2*len = 0
            let len = distances.len() as f64;
            let n = ((1.0 + (1.0 + 8.0 * len).sqrt()) / 2.0).floor() as usize;

            if n * (n - 1) / 2 != distances.len() {
                return Err(format!(
                    "Invalid distance matrix size: expected n*(n-1)/2, got {}",
                    distances.len()
                ));
            }

            // 转换为 f32
            let distances_f32: Vec<f32> = distances.iter().map(|&x| x as f32).collect();
            let dm = CompressedDistanceMatrix::from_distances(distances_f32, n);
            (dm, n)
        }
    };

    let threshold = threshold.unwrap_or(Value::INFINITY);

    // 2. 创建结果容器
    let mut result = RipserResult::new(num_points, max_dim, threshold);

    // 3. 创建二项式系数表
    // 需要支持 C(n, k)，其中 k <= max_dim + 2
    let max_k = (max_dim + 2).min(num_points);
    let binomial = BinomialCoeffTable::new(num_points, max_k);

    // 4. 枚举边（1-simplices）
    let edge_enumerator = EdgeEnumerator::from_compressed_matrix(&distance_matrix, threshold, &binomial);
    let edges = edge_enumerator.get_edges();

    // 5. 计算 0 维同调（连通分量）
    let dim_0_pairs = compute_dim_0_pairs(&edges, num_points, &binomial);
    result.set_persistence(0, dim_0_pairs);

    // 6. 计算高维同调（1, 2, ...）
    if max_dim >= 1 {
        // 构建完整的 simplex filtration
        let filtration = SimplexFiltration::build(
            &distance_matrix,
            max_dim,
            threshold,
            &binomial,
        );

        // 获取所有 simplex 的直径
        let all_diameters = filtration.get_diameters();

        // 创建 (维度, 组合索引) 到全局索引的映射
        // 关键：组合索引在不同维度之间不唯一，必须包含维度信息
        let mut combinatorial_to_global: std::collections::HashMap<(usize, Index), usize> =
            std::collections::HashMap::new();
        for simplex in &filtration.simplices {
            let dim = simplex.vertices.len() - 1;
            combinatorial_to_global.insert((dim, simplex.combinatorial_index), simplex.global_index);
        }

        // 计算各维度的同调
        // 关键：计算 H_d 需要归约 (d+1)-simplices 的边界
        for dim in 1..=max_dim {
            // 获取 (dim+1)-simplices（例如计算 H_1 需要 2-simplices）
            let higher_dim_simplices = filtration.get_simplices_by_dim(dim + 1);

            if higher_dim_simplices.is_empty() {
                continue;
            }

            // 构建 (dim+1)-simplices 的 DiameterEntry 数组和列索引到全局索引的映射
            let mut simplex_entries = Vec::new();
            let mut column_to_global = Vec::new();  // 列索引 → 全局索引

            for simplex in higher_dim_simplices {
                simplex_entries.push(DiameterEntry::new(simplex.diameter, simplex.combinatorial_index, 0));
                column_to_global.push(simplex.global_index);
            }


            // 计算持久性对：归约 (dim+1)-simplices 的边界到 dim-simplices
            let dim_pairs = compute_dim_pairs_with_mapping(
                &simplex_entries,
                dim + 1,  // 传入的是 (dim+1)-simplices 的维度
                num_points,
                &binomial,
                &all_diameters,
                &combinatorial_to_global,
                &column_to_global,  // 传递列索引映射
            );


            result.set_persistence(dim, dim_pairs);
        }
    }

    Ok(result)
}

/// 计算持久性对（带索引映射）
///
/// 这个版本使用 (维度, 组合索引) 到全局索引的映射来正确访问 diameters
fn compute_dim_pairs_with_mapping(
    simplices: &[DiameterEntry],
    dim: usize,
    n: usize,
    binomial: &BinomialCoeffTable,
    all_diameters: &[Value],
    combinatorial_to_global: &std::collections::HashMap<(usize, Index), usize>,
    column_to_global: &[usize],  // 矩阵列索引 → 全局索引
) -> Vec<(Value, Value)> {
    use super::cohomology::{BoundaryMatrix, SparseColumn, reduce_boundary_matrix};
    use super::simplex::SimplexBoundaryEnumerator;

    // 构建边界矩阵
    let mut boundary_matrix = BoundaryMatrix::with_capacity(simplices.len());


    for (_col_idx, simplex) in simplices.iter().enumerate() {
        // 计算边界
        let mut boundary_enum = SimplexBoundaryEnumerator::new(simplex.index, dim, n);
        let mut boundary_indices = Vec::new();

        // 使用 next_boundary 方法迭代
        while let Some(boundary_face) = boundary_enum.next_boundary(binomial) {
            // 将 (维度, 组合索引) 映射到全局索引
            // 边界的维度是 dim - 1
            let boundary_dim = dim - 1;
            if let Some(&global_idx) = combinatorial_to_global.get(&(boundary_dim, boundary_face.index)) {
                boundary_indices.push(global_idx as Index);
            }
        }

        // 添加到边界矩阵
        let boundary_column = SparseColumn::from_indices(boundary_indices);
        boundary_matrix.add_column(boundary_column);
    }

    // 执行矩阵归约
    let pairs = reduce_boundary_matrix(&mut boundary_matrix, all_diameters, column_to_global);

    // 转换为 (birth, death) 格式
    pairs
        .iter()
        .map(|pair| (pair.birth_diameter, pair.death_diameter))
        .collect()
}

/// 解析距离度量字符串
fn parse_metric(metric: &str) -> Result<Metric, String> {
    match metric.to_lowercase().as_str() {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "cityblock" => Ok(Metric::Manhattan),
        "chebyshev" | "chessboard" => Ok(Metric::Chebyshev),
        _ => Err(format!("Unknown metric: {}", metric)),
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 格式化持久性对为字符串（用于调试）
pub fn format_persistence(pairs: &[(Value, Value)]) -> String {
    let mut lines = Vec::new();
    for (i, (birth, death)) in pairs.iter().enumerate() {
        if death.is_infinite() {
            lines.push(format!("  {}: [{:.4}, ∞)", i, birth));
        } else {
            lines.push(format!("  {}: [{:.4}, {:.4})", i, birth, death));
        }
    }
    lines.join("\n")
}

/// 过滤持久性对（移除持久性太短的）
pub fn filter_persistence(
    pairs: &[(Value, Value)],
    min_persistence: Value,
) -> Vec<(Value, Value)> {
    pairs
        .iter()
        .filter(|(birth, death)| {
            if death.is_infinite() {
                true
            } else {
                death - birth >= min_persistence
            }
        })
        .copied()
        .collect()
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_ripser_simple() {
        // 3 个点的简单例子
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let result = compute_ripser(
            DistanceInput::Points(&points),
            1,
            None,
            "euclidean",
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.num_points, 3);
        assert_eq!(result.max_dim, 1);

        // 应该有 0 维和 1 维的持久性
        assert!(result.get_persistence(0).is_some());
        assert!(result.get_persistence(1).is_some());

        // 0 维：3 个点合并成 1 个分量
        let dim_0 = result.get_persistence(0).unwrap();
        assert_eq!(dim_0.len(), 3); // 2 个有限对 + 1 个无穷对
    }

    #[test]
    fn test_compute_ripser_with_threshold() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0], // 远离的点
        ];

        let result = compute_ripser(
            DistanceInput::Points(&points),
            1,
            Some(2.0), // 阈值 2.0
            "euclidean",
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.num_points, 4);

        // 由于阈值限制，远离的点不会连接
        let dim_0 = result.get_persistence(0).unwrap();
        // 应该有 2 个连通分量（前 3 个点 + 最后 1 个点）
        let infinite_pairs = dim_0.iter().filter(|(_, d)| d.is_infinite()).count();
        assert_eq!(infinite_pairs, 2);
    }

    #[test]
    fn test_parse_metric() {
        assert!(matches!(
            parse_metric("euclidean"),
            Ok(Metric::Euclidean)
        ));
        assert!(matches!(
            parse_metric("manhattan"),
            Ok(Metric::Manhattan)
        ));
        assert!(matches!(
            parse_metric("chebyshev"),
            Ok(Metric::Chebyshev)
        ));
        assert!(parse_metric("unknown").is_err());
    }

    #[test]
    fn test_filter_persistence() {
        let pairs = vec![
            (0.0, 0.1),
            (0.0, 1.0),
            (0.5, 0.6),
            (0.0, Value::INFINITY),
        ];

        let filtered = filter_persistence(&pairs, 0.5);

        // 只保留持久性 >= 0.5 的
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&(0.0, 1.0)));
        assert!(filtered.contains(&(0.0, Value::INFINITY)));
    }

    #[test]
    fn test_distance_input_compressed() {
        // 3 个点的压缩距离矩阵
        // 点: (0,0), (1,0), (0,1)
        // 距离: d(0,1)=1.0, d(0,2)=1.0, d(1,2)=sqrt(2)
        let distances = vec![1.0, 1.0, 2.0_f64.sqrt()];

        let result = compute_ripser(
            DistanceInput::Compressed(&distances),
            1,
            None,
            "euclidean", // metric 对压缩矩阵无影响
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.num_points, 3);
    }

    #[test]
    fn test_ripser_result_methods() {
        let mut result = RipserResult::new(5, 2, 1.0);

        assert_eq!(result.num_points, 5);
        assert_eq!(result.max_dim, 2);
        assert_eq!(result.threshold, 1.0);

        // 设置持久性对
        result.set_persistence(0, vec![(0.0, 1.0), (0.0, Value::INFINITY)]);
        result.set_persistence(1, vec![(0.5, 1.5)]);

        assert_eq!(result.get_persistence(0).unwrap().len(), 2);
        assert_eq!(result.get_persistence(1).unwrap().len(), 1);
        assert_eq!(result.total_pairs(), 3);
    }
}
