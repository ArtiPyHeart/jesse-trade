/*!
 * 上同调计算
 *
 * 这个模块实现持久性上同调的计算，包括：
 * - UnionFind 数据结构（用于 0 维同调）
 * - 0 维同调计算（连通分量）
 * - 高维同调计算（矩阵归约）
 *
 * # 算法概览
 *
 * ## 0 维同调（连通分量）
 *
 * 使用 Kruskal 最小生成树算法 + Union-Find 数据结构：
 * 1. 按直径降序遍历边
 * 2. 对每条边，如果连接不同分量，则记录持久性对
 * 3. 较年轻的分量（较大的 birth）先死（elder rule）
 *
 * ## 高维同调（矩阵归约）
 *
 * 使用稀疏矩阵的列归约算法：
 * 1. 计算边界矩阵
 * 2. 通过 pivot 消元找到持久性对
 * 3. 使用 Z/2Z 系数域（简化计算）
 *
 * # 使用示例
 *
 * ```ignore
 * // 计算 0 维同调
 * let edges = edge_enumerator.get_edges();
 * let pairs = compute_dim_0_pairs(edges, n_vertices, &binomial);
 * ```
 */

use super::super::types::{DiameterEntry, Index, Value};
use super::binomial::BinomialCoeffTable;
use super::simplex::get_edge_vertices;

// ============================================================================
// UnionFind - 并查集（支持 birth time 追踪）
// ============================================================================

/// UnionFind 数据结构
///
/// 扩展的并查集，额外追踪每个分量的 birth time。
/// 用于计算 0 维持久性同调（连通分量）。
///
/// # 特性
///
/// - **路径压缩**: `find` 操作时扁平化树结构
/// - **按秩合并**: `union` 操作时合并到更高的树
/// - **Birth tracking**: 记录每个分量的最早 birth 时间
/// - **Elder rule**: 合并时，较年轻的分量先死
///
/// # 示例
///
/// ```ignore
/// let mut uf = UnionFind::new(5);
/// uf.union(0, 1, 1.0);  // 合并顶点 0 和 1，death time = 1.0
/// uf.union(1, 2, 2.0);  // 合并顶点 1 和 2，death time = 2.0
/// assert_eq!(uf.find(0), uf.find(2));  // 0 和 2 在同一分量
/// ```
pub struct UnionFind {
    /// 父节点索引（parent[i] = i 表示 i 是根）
    parent: Vec<usize>,

    /// 树的秩（用于按秩合并优化）
    rank: Vec<u8>,

    /// 分量的 birth time（顶点权重，默认为 0）
    birth: Vec<Value>,

    /// birth 顶点的索引（用于追踪哪个顶点创建了分量）
    birth_vertex: Vec<usize>,
}

impl UnionFind {
    /// 创建新的 UnionFind
    ///
    /// # Arguments
    ///
    /// * `n` - 元素数量
    ///
    /// # Returns
    ///
    /// 初始化的 UnionFind，每个元素独立成一个集合
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth: vec![0.0; n],
            birth_vertex: (0..n).collect(),
        }
    }

    /// 查找元素所属的集合（根节点）
    ///
    /// 使用路径压缩优化
    ///
    /// # Arguments
    ///
    /// * `x` - 元素索引
    ///
    /// # Returns
    ///
    /// 根节点索引
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            // 路径压缩：将 x 直接连接到根节点
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// 合并两个集合并返回较年轻分量的 birth 顶点
    ///
    /// 实现 "elder rule"：较年轻的分量（较大的 birth）先死
    ///
    /// # Arguments
    ///
    /// * `x` - 第一个元素
    /// * `y` - 第二个元素
    /// * `death` - 合并时的 death time（当前边的直径）
    ///
    /// # Returns
    ///
    /// 较年轻分量的 birth 顶点索引
    pub fn union(&mut self, x: usize, y: usize, _death: Value) -> usize {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            // 已经在同一集合
            return self.birth_vertex[root_x];
        }

        // 按秩合并
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
            // Elder rule: 返回较年轻的分量
            if self.birth[root_x] > self.birth[root_y] {
                self.birth_vertex[root_y] = self.birth_vertex[root_x];
                self.birth_vertex[root_x]
            } else {
                self.birth_vertex[root_y]
            }
        } else {
            self.parent[root_y] = root_x;
            if self.rank[root_x] == self.rank[root_y] {
                self.rank[root_x] += 1;
            }
            // Elder rule: 返回较年轻的分量
            if self.birth[root_y] > self.birth[root_x] {
                self.birth_vertex[root_x] = self.birth_vertex[root_y];
                self.birth_vertex[root_y]
            } else {
                self.birth_vertex[root_x]
            }
        }
    }

    /// 获取分量的 birth time
    ///
    /// # Arguments
    ///
    /// * `x` - 元素索引
    ///
    /// # Returns
    ///
    /// 该分量的 birth time
    pub fn get_birth(&mut self, x: usize) -> Value {
        let root = self.find(x);
        self.birth[self.birth_vertex[root]]
    }

    /// 设置顶点的 birth time
    ///
    /// # Arguments
    ///
    /// * `x` - 顶点索引
    /// * `birth` - birth time
    pub fn set_birth(&mut self, x: usize, birth: Value) {
        self.birth[x] = birth;
    }

    /// 获取所有独立分量的根节点
    ///
    /// # Returns
    ///
    /// 根节点索引的向量
    pub fn get_roots(&mut self) -> Vec<usize> {
        let n = self.parent.len();
        let mut roots = Vec::new();

        for i in 0..n {
            if self.find(i) == i {
                roots.push(i);
            }
        }

        roots
    }
}

// ============================================================================
// 0 维同调计算（连通分量）
// ============================================================================

/// 计算 0 维持久性同调（连通分量）
///
/// 使用 Kruskal 最小生成树算法 + Union-Find 数据结构。
///
/// # 算法流程
///
/// 1. 初始化 UnionFind（每个顶点独立成一个分量）
/// 2. 按直径**降序**遍历边（从大到小）
/// 3. 对每条边 {i, j}:
///    - 如果 i 和 j 在不同分量，记录持久性对并合并
///    - 使用 elder rule：较年轻的分量先死
/// 4. 最后剩余的分量具有无穷长的持久性
///
/// # Arguments
///
/// * `edges` - 边的列表（已按直径升序排序）
/// * `n` - 顶点数量
/// * `binomial` - 二项式系数表（用于解码边）
///
/// # Returns
///
/// 持久性对的向量 [(birth, death), ...]
///
/// # 示例
///
/// ```ignore
/// let edges = edge_enumerator.get_edges();
/// let pairs = compute_dim_0_pairs(edges, 100, &binomial);
///
/// for (birth, death) in pairs {
///     println!("Connected component: [{}, {})", birth, death);
/// }
/// ```
pub fn compute_dim_0_pairs(
    edges: &[DiameterEntry],
    n: usize,
    binomial: &BinomialCoeffTable,
) -> Vec<(Value, Value)> {
    let mut uf = UnionFind::new(n);
    let mut pairs = Vec::new();

    // 按直径升序遍历边（从小到大）
    // 这是filtration的正确顺序：边按直径增大的顺序出现
    // 当两个分量首次通过边连接时，记录death时间为该边的直径
    for edge in edges.iter() {
        let (i, j) = get_edge_vertices(edge.index, binomial);

        let root_i = uf.find(i);
        let root_j = uf.find(j);

        if root_i != root_j {
            // 连接两个不同的分量
            let birth_vertex = uf.union(i, j, edge.diameter);
            let birth = uf.get_birth(birth_vertex);
            let death = edge.diameter;

            // 只记录有意义的持久性对（death > birth）
            if death > birth {
                pairs.push((birth, death));
            }
        }
    }

    // 添加无穷长的连通分量（从未死亡的分量）
    let roots = uf.get_roots();
    for root in roots {
        let birth = uf.get_birth(root);
        pairs.push((birth, Value::INFINITY));
    }

    pairs
}

// ============================================================================
// 稀疏矩阵 - 压缩稀疏列（CSC）格式
// ============================================================================

/// 稀疏列向量（用于边界矩阵）
///
/// 使用有序向量存储非零元素的索引。
/// 在 Z/2Z 系数域中，所有非零元素都是 1，因此只需存储索引。
///
/// # 示例
///
/// ```ignore
/// let mut col = SparseColumn::new();
/// col.add(5);   // 添加元素 5
/// col.add(3);   // 添加元素 3
/// col.add(5);   // 再次添加 5（在 Z/2Z 中会删除）
/// assert_eq!(col.get_pivot(), Some(3));  // pivot 是最大的元素
/// ```
#[derive(Debug, Clone)]
pub struct SparseColumn {
    /// 非零元素的索引（降序排列）
    indices: Vec<Index>,
}

impl SparseColumn {
    /// 创建空的稀疏列
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    /// 从索引向量创建稀疏列
    ///
    /// # Arguments
    ///
    /// * `indices` - 索引向量（会自动排序并去重）
    pub fn from_indices(mut indices: Vec<Index>) -> Self {
        // 排序并去重
        indices.sort_unstable();
        indices.reverse();  // 降序排列
        indices.dedup();
        Self { indices }
    }

    /// 添加一个元素（Z/2Z 域中的加法）
    ///
    /// 如果元素已存在，则删除（因为 1 + 1 = 0 in Z/2Z）
    ///
    /// # Arguments
    ///
    /// * `index` - 要添加的索引
    pub fn add(&mut self, index: Index) {
        // 在降序列表中找到插入位置
        match self.indices.binary_search_by(|&x| x.cmp(&index).reverse()) {
            Ok(pos) => {
                // 已存在，删除（Z/2Z 中 1 + 1 = 0）
                self.indices.remove(pos);
            }
            Err(pos) => {
                // 不存在，插入
                self.indices.insert(pos, index);
            }
        }
    }

    /// 添加另一列的所有元素（Z/2Z 域中的加法）
    ///
    /// # Arguments
    ///
    /// * `other` - 要添加的列
    pub fn add_column(&mut self, other: &SparseColumn) {
        for &index in &other.indices {
            self.add(index);
        }
    }

    /// 获取 pivot（最大的索引）
    ///
    /// # Returns
    ///
    /// 如果列非空，返回 Some(pivot)；否则返回 None
    pub fn get_pivot(&self) -> Option<Index> {
        self.indices.first().copied()
    }

    /// 检查列是否为空
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// 获取列中元素的数量
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// 获取所有索引的引用
    pub fn indices(&self) -> &[Index] {
        &self.indices
    }
}

impl Default for SparseColumn {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 边界矩阵 - 压缩稀疏列格式
// ============================================================================

/// 边界矩阵（用于矩阵归约）
///
/// 存储简单复形的边界算子矩阵。
/// 对于 k-simplex，其边界是 (k-1)-simplices 的集合。
///
/// # 示例
///
/// ```ignore
/// let mut boundary_matrix = BoundaryMatrix::new();
/// boundary_matrix.add_column(simplex_index, boundary_column);
/// ```
pub struct BoundaryMatrix {
    /// 列向量（indexed by simplex index）
    columns: Vec<SparseColumn>,
}

impl BoundaryMatrix {
    /// 创建新的边界矩阵
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// 创建指定容量的边界矩阵
    ///
    /// # Arguments
    ///
    /// * `capacity` - 预分配的列数
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            columns: Vec::with_capacity(capacity),
        }
    }

    /// 添加一列
    ///
    /// # Arguments
    ///
    /// * `column` - 要添加的稀疏列
    pub fn add_column(&mut self, column: SparseColumn) {
        self.columns.push(column);
    }

    /// 获取列的可变引用
    ///
    /// # Arguments
    ///
    /// * `index` - 列索引
    pub fn get_column_mut(&mut self, index: usize) -> Option<&mut SparseColumn> {
        self.columns.get_mut(index)
    }

    /// 获取列的引用
    ///
    /// # Arguments
    ///
    /// * `index` - 列索引
    pub fn get_column(&self, index: usize) -> Option<&SparseColumn> {
        self.columns.get(index)
    }

    /// 获取矩阵的列数
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }
}

impl Default for BoundaryMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 矩阵归约算法
// ============================================================================

/// 持久性对（birth-death pair）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    /// Birth simplex index
    pub birth: Index,
    /// Death simplex index (或 Index::MAX 表示无穷)
    pub death: Index,
    /// Birth 时的直径
    pub birth_diameter: Value,
    /// Death 时的直径
    pub death_diameter: Value,
}

/// Pivot 追踪表
///
/// 用于记录每个 pivot 对应的列索引，避免重复计算
struct PivotTracker {
    /// pivot_to_column[pivot] = column index
    pivot_to_column: Vec<Option<usize>>,
}

impl PivotTracker {
    /// 创建新的 pivot 追踪表
    ///
    /// # Arguments
    ///
    /// * `max_pivot` - 可能的最大 pivot 值
    fn new(max_pivot: usize) -> Self {
        Self {
            pivot_to_column: vec![None; max_pivot + 1],
        }
    }

    /// 设置 pivot 对应的列
    ///
    /// # Arguments
    ///
    /// * `pivot` - pivot 值
    /// * `column` - 列索引
    fn set(&mut self, pivot: Index, column: usize) {
        if (pivot as usize) < self.pivot_to_column.len() {
            self.pivot_to_column[pivot as usize] = Some(column);
        }
    }

    /// 获取 pivot 对应的列
    ///
    /// # Arguments
    ///
    /// * `pivot` - pivot 值
    fn get(&self, pivot: Index) -> Option<usize> {
        if (pivot as usize) < self.pivot_to_column.len() {
            self.pivot_to_column[pivot as usize]
        } else {
            None
        }
    }

    /// 清除 pivot
    ///
    /// # Arguments
    ///
    /// * `pivot` - pivot 值
    #[allow(dead_code)]
    fn clear(&mut self, pivot: Index) {
        if (pivot as usize) < self.pivot_to_column.len() {
            self.pivot_to_column[pivot as usize] = None;
        }
    }
}

/// 矩阵归约（标准算法）
///
/// 使用列归约算法计算持久性同调。
///
/// # 算法流程
///
/// 1. 对每一列，找到其 pivot（最大的非零索引）
/// 2. 如果 pivot 已被使用，则用前一列消元
/// 3. 重复直到 pivot 唯一或列变为零
/// 4. 记录持久性对：(pivot, column)
///
/// # Arguments
///
/// * `boundary_matrix` - 边界矩阵（会被修改）
/// * `diameters` - 每个 simplex 的直径
///
/// # Returns
///
/// 持久性对的向量
pub fn reduce_boundary_matrix(
    boundary_matrix: &mut BoundaryMatrix,
    diameters: &[Value],
    column_to_global: &[usize],  // 矩阵列索引 → 全局 simplex 索引
) -> Vec<PersistencePair> {
    let n_columns = boundary_matrix.num_columns();
    let mut pairs = Vec::new();

    // 创建 pivot 追踪表
    // 修复（基于 Codex 分析）：用 diameters.len() - 1 作为 max_pivot
    // 因为 pivot 是全局边界单纯形索引（范围 [0, diameters.len())），
    // 可能远大于列数 n_columns（特别是稀疏 filtration 中 #edges >> #triangles）
    let max_pivot = if diameters.len() > 0 { diameters.len() - 1 } else { 0 };
    let mut pivot_tracker = PivotTracker::new(max_pivot);


    // 对每一列进行归约
    for j in 0..n_columns {
        // 获取当前列的可变引用
        let mut working_column = boundary_matrix.columns[j].clone();

        // 持续消元直到 pivot 唯一或列变为零
        loop {
            let pivot = match working_column.get_pivot() {
                Some(p) => p,
                None => break,  // 列为零，停止
            };


            // 检查 pivot 是否已被使用
            match pivot_tracker.get(pivot) {
                Some(prev_column) => {
                    // Pivot 已被使用，用前一列消元
                    let prev_col = boundary_matrix.columns[prev_column].clone();
                    working_column.add_column(&prev_col);
                }
                None => {
                    // Pivot 唯一，记录并停止
                    pivot_tracker.set(pivot, j);

                    // 记录持久性对
                    let birth = pivot;  // birth 是边界面的全局索引（pivot）
                    let death = column_to_global[j];  // death 是当前列对应的全局索引
                    let birth_diameter = diameters[birth as usize];
                    let death_diameter = diameters[death];

                    // 记录持久性对（包括零长度的，用于检测退化情况）
                    if birth_diameter <= death_diameter {
                        pairs.push(PersistencePair {
                            birth,
                            death: death as Index,
                            birth_diameter,
                            death_diameter,
                        });
                    }

                    break;
                }
            }
        }

        // 更新边界矩阵中的列
        boundary_matrix.columns[j] = working_column;
    }

    pairs
}

// ============================================================================
// 高维同调计算
// ============================================================================

/// 计算指定维度的持久性同调
///
/// # Arguments
///
/// * `simplices` - 简单复形列表（DiameterEntry）
/// * `dim` - 要计算的维度
/// * `n` - 顶点数量（最大顶点索引 + 1）
/// * `binomial` - 二项式系数表
///
/// # Returns
///
/// 持久性对的向量
pub fn compute_dim_pairs(
    simplices: &[DiameterEntry],
    dim: usize,
    n: usize,
    binomial: &BinomialCoeffTable,
) -> Vec<(Value, Value)> {
    use super::simplex::SimplexBoundaryEnumerator;

    // 构建边界矩阵
    let mut boundary_matrix = BoundaryMatrix::with_capacity(simplices.len());
    let mut diameters = Vec::with_capacity(simplices.len());

    for simplex in simplices {
        // 计算边界
        let mut boundary_enum = SimplexBoundaryEnumerator::new(simplex.index, dim, n);
        let mut boundary_indices = Vec::new();

        // 使用 next_boundary 方法迭代
        while let Some(boundary_face) = boundary_enum.next_boundary(binomial) {
            boundary_indices.push(boundary_face.index);
        }

        // 添加到边界矩阵
        let boundary_column = SparseColumn::from_indices(boundary_indices);
        boundary_matrix.add_column(boundary_column);
        diameters.push(simplex.diameter);
    }

    // 执行矩阵归约
    // compute_dim_0_pairs 使用局部的 diameters 数组，列索引就是数组索引（identity mapping）
    let column_to_global: Vec<usize> = (0..diameters.len()).collect();
    let pairs = reduce_boundary_matrix(&mut boundary_matrix, &diameters, &column_to_global);

    // 转换为 (birth, death) 格式
    let result: Vec<(Value, Value)> = pairs
        .iter()
        .map(|pair| (pair.birth_diameter, pair.death_diameter))
        .collect();

    result
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);

        // 初始状态：每个元素独立
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 2);

        // 合并 0 和 1
        uf.union(0, 1, 1.0);
        assert_eq!(uf.find(0), uf.find(1));

        // 合并 2 和 3
        uf.union(2, 3, 2.0);
        assert_eq!(uf.find(2), uf.find(3));

        // 0 和 2 仍然分离
        assert_ne!(uf.find(0), uf.find(2));

        // 合并两个分量
        uf.union(1, 2, 3.0);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new(4);

        // 创建链：0 -> 1 -> 2 -> 3
        uf.union(0, 1, 1.0);
        uf.union(1, 2, 2.0);
        uf.union(2, 3, 3.0);

        // 查找会触发路径压缩
        let root = uf.find(0);
        assert_eq!(root, uf.find(3));

        // 路径压缩后，0 应该直接指向根
        assert_eq!(uf.parent[0], root);
    }

    #[test]
    fn test_union_find_birth_tracking() {
        let mut uf = UnionFind::new(3);

        // 设置 birth time
        uf.set_birth(0, 0.0);
        uf.set_birth(1, 1.0);
        uf.set_birth(2, 0.5);

        assert_eq!(uf.get_birth(0), 0.0);
        assert_eq!(uf.get_birth(1), 1.0);
        assert_eq!(uf.get_birth(2), 0.5);

        // 合并后，birth 应该是较年轻的那个
        uf.union(0, 1, 2.0);
        let root = uf.find(0);
        assert_eq!(uf.get_birth(root), 1.0);  // 顶点 1 较年轻
    }

    #[test]
    fn test_compute_dim_0_simple() {
        use super::super::binomial::BinomialCoeffTable;
        use super::super::simplex::get_edge_index;

        let binomial = BinomialCoeffTable::new(10, 5);

        // 创建一个简单的例子：3 个顶点，2 条边
        // {0, 1} 距离 1.0
        // {1, 2} 距离 2.0
        let edges = vec![
            DiameterEntry::new(1.0, get_edge_index(1, 0, &binomial), 0),
            DiameterEntry::new(2.0, get_edge_index(2, 1, &binomial), 0),
        ];

        let pairs = compute_dim_0_pairs(&edges, 3, &binomial);

        // 应该有 3 个持久性对：
        // - 2 个有限对（分量合并）
        // - 1 个无穷对（最终的单一分量）
        assert_eq!(pairs.len(), 3);

        // 检查是否有无穷对
        assert!(pairs.iter().any(|(_, death)| death.is_infinite()));

        // 有限对的 death 应该是边的直径
        let finite_pairs: Vec<_> = pairs.iter().filter(|(_, d)| d.is_finite()).collect();
        assert_eq!(finite_pairs.len(), 2);
    }

    #[test]
    fn test_compute_dim_0_disconnected() {
        use super::super::binomial::BinomialCoeffTable;
        use super::super::simplex::get_edge_index;

        let binomial = BinomialCoeffTable::new(10, 5);

        // 4 个顶点，只有 1 条边 {0, 1}
        // 形成 2 个分量：{0, 1} 和 {2}, {3}
        let edges = vec![DiameterEntry::new(1.0, get_edge_index(1, 0, &binomial), 0)];

        let pairs = compute_dim_0_pairs(&edges, 4, &binomial);

        // 应该有 4 个持久性对：
        // - 1 个有限对（0 和 1 合并）
        // - 3 个无穷对（3 个最终分量）
        assert_eq!(pairs.len(), 4);

        let infinite_pairs = pairs.iter().filter(|(_, d)| d.is_infinite()).count();
        assert_eq!(infinite_pairs, 3);
    }

    #[test]
    fn test_union_find_get_roots() {
        let mut uf = UnionFind::new(5);

        // 初始：5 个独立分量
        let roots = uf.get_roots();
        assert_eq!(roots.len(), 5);

        // 合并一些元素
        uf.union(0, 1, 1.0);
        uf.union(2, 3, 2.0);

        let roots = uf.get_roots();
        assert_eq!(roots.len(), 3);  // {0,1}, {2,3}, {4}
    }

    #[test]
    fn test_elder_rule() {
        let mut uf = UnionFind::new(3);

        // 设置不同的 birth time
        uf.set_birth(0, 0.0);  // 较老
        uf.set_birth(1, 1.0);  // 较年轻

        // 合并时，较年轻的先死
        let younger = uf.union(0, 1, 2.0);

        // younger 应该是顶点 1（birth = 1.0 > 0.0）
        assert_eq!(younger, 1);
    }

    // ========================================================================
    // 稀疏矩阵测试
    // ========================================================================

    #[test]
    fn test_sparse_column_basic() {
        let mut col = SparseColumn::new();
        assert!(col.is_empty());
        assert_eq!(col.get_pivot(), None);

        // 添加元素
        col.add(5);
        col.add(3);
        col.add(7);

        assert_eq!(col.len(), 3);
        assert_eq!(col.get_pivot(), Some(7));  // pivot 是最大的元素
    }

    #[test]
    fn test_sparse_column_z2_addition() {
        let mut col = SparseColumn::new();

        // 添加元素
        col.add(5);
        col.add(3);
        assert_eq!(col.len(), 2);

        // 再次添加 5（Z/2Z 中 1 + 1 = 0）
        col.add(5);
        assert_eq!(col.len(), 1);
        assert_eq!(col.get_pivot(), Some(3));

        // 再次添加 3，列变为空
        col.add(3);
        assert!(col.is_empty());
        assert_eq!(col.get_pivot(), None);
    }

    #[test]
    fn test_sparse_column_from_indices() {
        let indices = vec![5, 3, 7, 3, 5];  // 包含重复元素
        let col = SparseColumn::from_indices(indices);

        // 应该去重并降序排列
        assert_eq!(col.len(), 3);
        assert_eq!(col.indices(), &[7, 5, 3]);
        assert_eq!(col.get_pivot(), Some(7));
    }

    #[test]
    fn test_sparse_column_add_column() {
        let mut col1 = SparseColumn::new();
        col1.add(5);
        col1.add(3);

        let mut col2 = SparseColumn::new();
        col2.add(7);
        col2.add(5);  // 与 col1 重复

        // 添加 col2 到 col1
        col1.add_column(&col2);

        // 结果应该是 {3, 7}（5 被抵消）
        assert_eq!(col1.len(), 2);
        assert_eq!(col1.get_pivot(), Some(7));
        assert!(col1.indices().contains(&3));
        assert!(col1.indices().contains(&7));
    }

    #[test]
    fn test_boundary_matrix_basic() {
        let mut boundary_matrix = BoundaryMatrix::new();
        assert_eq!(boundary_matrix.num_columns(), 0);

        // 添加列
        let mut col1 = SparseColumn::new();
        col1.add(0);
        col1.add(1);
        boundary_matrix.add_column(col1);

        assert_eq!(boundary_matrix.num_columns(), 1);
        assert_eq!(boundary_matrix.get_column(0).unwrap().len(), 2);
    }

    #[test]
    fn test_reduce_boundary_matrix_simple() {
        // 简单的例子：三角形
        // 顶点: 0, 1, 2
        // 边: {0,1}, {0,2}, {1,2}
        // 三角形: {0,1,2}
        //
        // 边界矩阵（边的边界是顶点）:
        // Edge {0,1}: boundary = {0, 1}
        // Edge {0,2}: boundary = {0, 2}
        // Edge {1,2}: boundary = {1, 2}

        let mut boundary_matrix = BoundaryMatrix::new();

        // 添加边的边界
        boundary_matrix.add_column(SparseColumn::from_indices(vec![0, 1]));
        boundary_matrix.add_column(SparseColumn::from_indices(vec![0, 2]));
        boundary_matrix.add_column(SparseColumn::from_indices(vec![1, 2]));

        // 设置直径（按升序）
        let diameters = vec![1.0, 1.5, 2.0];

        // Identity mapping for test
        let column_to_global: Vec<usize> = (0..diameters.len()).collect();

        // 执行归约
        let pairs = reduce_boundary_matrix(&mut boundary_matrix, &diameters, &column_to_global);

        // 检查结果
        // 应该没有持久性对（因为所有边的 birth 都是 0）
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_sparse_column_ordering() {
        let mut col = SparseColumn::new();
        col.add(3);
        col.add(7);
        col.add(1);
        col.add(5);

        // 应该按降序排列
        assert_eq!(col.indices(), &[7, 5, 3, 1]);
        assert_eq!(col.get_pivot(), Some(7));
    }

    #[test]
    fn test_pivot_tracker() {
        let mut tracker = PivotTracker::new(10);

        // 初始状态
        assert_eq!(tracker.get(5), None);

        // 设置 pivot
        tracker.set(5, 3);
        assert_eq!(tracker.get(5), Some(3));

        // 清除 pivot
        tracker.clear(5);
        assert_eq!(tracker.get(5), None);
    }
}
