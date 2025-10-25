# Ripser 算法原理与实现指南

**版本**: 迭代 0（研究与架构设计）
**日期**: 2025年
**基于**: ripser.h (1857行, C++ 实现)

---

## 📖 目录

1. [算法概述](#算法概述)
2. [核心概念](#核心概念)
3. [数据结构](#数据结构)
4. [核心算法](#核心算法)
5. [并行化策略](#并行化策略)
6. [Rust 实现路线图](#rust-实现路线图)

---

## 算法概述

### 什么是 Ripser？

Ripser 是计算 Vietoris-Rips 持久同调的高效算法，用于拓扑数据分析（TDA）。给定一个点云或距离矩阵，Ripser 可以计算其拓扑特征（如连通分量、环、空洞）如何随着距离阈值变化而出现和消失。

### 核心思想

1. **Vietoris-Rips 复形**: 从距离矩阵构建简单复形（simplex）
   - 0-simplex (顶点): 所有点
   - 1-simplex (边): 距离 ≤ 阈值的点对
   - k-simplex: 所有点对距离 ≤ 阈值的点集合

2. **持久同调**: 跟踪拓扑特征的生成（birth）和消亡（death）
   - H0: 连通分量
   - H1: 环/循环
   - H2: 空洞/球面
   - Hk: k维"洞"

3. **矩阵归约**: 通过稀疏矩阵归约算法计算持久性
   - 使用上同调（cohomology）而非同调（homology）
   - 基于 pivot 的增量归约
   - 并行化处理高维同调

---

## 核心概念

### 1. 组合编码（Combinatorial Encoding）

**问题**: 如何高效存储和访问简单复形？

**解决方案**: 将 k-simplex 编码为单个整数索引

#### 编码方案

对于 k-simplex {v0, v1, ..., vk}（其中 v0 < v1 < ... < vk）：

```
index = C(vk, k+1) + C(vk-1, k) + ... + C(v1, 2) + v0
```

其中 C(n, k) 是二项式系数。

**示例**（n=5个顶点）:
- 边 {0, 2}:  index = C(2, 2) + 0 = 1 + 0 = 1
- 边 {1, 3}:  index = C(3, 2) + 1 = 3 + 1 = 4
- 三角形 {0, 2, 4}: index = C(4, 3) + C(2, 2) + 0 = 4 + 1 + 0 = 5

**优势**:
- 紧凑存储：O(1) 空间表示任意维度的 simplex
- 快速编解码：通过二项式系数表查表
- 字典序排序：索引天然有序

### 2. 过滤（Filtration）

**定义**: 简单复形按直径（diameter）排序的序列

```
∅ ⊆ K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ
```

**直径**: k-simplex 的直径 = 其所有边的最大距离

```rust
fn compute_diameter(vertices: &[usize], dist: &DistanceMatrix) -> f32 {
    let mut max_dist = 0.0;
    for i in 0..vertices.len() {
        for j in 0..i {
            max_dist = max_dist.max(dist.get(vertices[i], vertices[j]));
        }
    }
    max_dist
}
```

### 3. 持久性对（Persistence Pairs）

每个拓扑特征对应一个持久性对 (birth, death)：
- **birth**: 特征首次出现的直径
- **death**: 特征消失的直径
- **persistence**: death - birth（持续时间）

**示例**:
- 连通分量 (0.0, 1.5): 在距离0时出现，1.5时与其他分量合并
- 环 (2.0, 3.5): 在距离2.0时形成，3.5时被填充

---

## 数据结构

### 1. 二项式系数表（Binomial Coefficient Table）

**目的**: 加速组合编码的计算

**结构**:
```rust
pub struct BinomialCoeffTable {
    // B[k][n] = C(n, k)
    table: Vec<Vec<i64>>,
}
```

**构建**（动态规划）:
```
C(n, k) = C(n-1, k-1) + C(n-1, k)
C(n, 0) = 1
C(n, n) = 1
```

**复杂度**:
- 空间: O(n * k)
- 构建: O(n * k)
- 查询: O(1)

**Rust 实现要点**:
- 使用 i64 避免溢出
- 检查 `max_simplex_index` 溢出
- 预分配矩阵大小

### 2. 距离矩阵（Distance Matrix）

#### 稠密矩阵（Compressed Distance Matrix）

**目的**: 节省内存，只存储上三角或下三角

**结构**（下三角）:
```rust
pub struct CompressedDistanceMatrix {
    diagonal: Vec<f32>,     // n个对角元素
    distances: Vec<f32>,    // n*(n-1)/2 个下三角元素
    rows: Vec<*mut f32>,    // 指向每行起始位置的指针
}
```

**索引计算**（下三角）:
```
对于 (i, j) 其中 i > j:
    offset = i*(i-1)/2 + j
```

**内存布局**:
```
n=5 的下三角矩阵:
  0   1   2   3   4
0 d00  -   -   -   -
1 d10 d11  -   -   -
2 d20 d21 d22  -   -
3 d30 d31 d32 d33  -
4 d40 d41 d42 d43 d44

存储: [d10, d20, d21, d30, d31, d32, d40, d41, d42, d43]
```

#### 稀疏矩阵（Sparse Distance Matrix）

**目的**: 处理大规模稀疏数据或使用距离阈值

**结构**:
```rust
pub struct SparseDistanceMatrix {
    // neighbors[i] = [(邻居索引, 距离), ...]
    neighbors: Vec<Vec<(usize, f32)>>,
    vertex_births: Vec<f32>,  // 顶点权重（可选）
}
```

**特点**:
- COO（Coordinate）格式输入
- 邻接表存储
- 支持距离阈值过滤
- 自动对称化

### 3. DiameterEntry（直径条目）

**核心数据结构**，贯穿整个算法：

```rust
pub struct DiameterEntry {
    diameter: f32,      // 过滤值
    index: i64,         // simplex 索引
    coefficient: u16,   // 系数（用于 Z/pZ）
}
```

**用途**:
- 边枚举: 存储 (距离, 边索引)
- 工作队列: 优先队列按直径排序
- 矩阵条目: 稀疏矩阵的条目

**排序规则**:
1. 按直径升序
2. 直径相同则按索引升序

### 4. 压缩稀疏矩阵（Compressed Sparse Matrix）

**目的**: 存储归约过程中的矩阵列

**结构**:
```rust
pub struct CompressedSparseMatrix {
    // 每列是一个 Vec<DiameterEntry>
    // 使用原子指针支持并发访问
    columns: Vec<AtomicPtr<Vec<DiameterEntry>>>,
}
```

**特点**:
- 列压缩格式（CSC）
- 原子操作支持并发
- 延迟分配（列按需创建）

---

## 核心算法

### 1. 边枚举（Edge Enumeration）

**目标**: 生成所有距离 ≤ 阈值的边

**算法**（稠密矩阵）:
```rust
fn get_edges(dist: &CompressedDistanceMatrix, threshold: f32) -> Vec<DiameterEntry> {
    let mut edges = Vec::new();
    let n = dist.size();

    for i in 0..n {
        for j in 0..i {
            let d = dist.get(i, j);
            if d <= threshold {
                let index = binomial_coeff(i, 2) + j;
                edges.push(DiameterEntry::new(d, index, 0));
            }
        }
    }

    edges.sort();  // 按直径排序
    edges
}
```

**算法**（稀疏矩阵）:
```rust
fn get_edges(dist: &SparseDistanceMatrix) -> Vec<DiameterEntry> {
    let mut edges = Vec::new();

    for i in 0..dist.size() {
        for &(j, d) in &dist.neighbors[i] {
            if i > j {  // 避免重复
                let index = get_edge_index(i, j);
                edges.push(DiameterEntry::new(d, index, 0));
            }
        }
    }

    edges.sort();
    edges
}
```

**复杂度**:
- 稠密: O(n² log n)
- 稀疏: O(m log m)，m = 边数

### 2. Simplex 编解码

#### 编码：顶点 → 索引

**算法**（边）:
```rust
fn get_edge_index(i: usize, j: usize, binomial: &BinomialCoeffTable) -> i64 {
    let (max, min) = if i > j { (i, j) } else { (j, i) };
    binomial.get(max, 2) + min as i64
}
```

#### 解码：索引 → 顶点

**算法**（通用）:
```rust
fn get_simplex_vertices(
    mut index: i64,
    dim: usize,
    mut n: usize,
    binomial: &BinomialCoeffTable,
) -> Vec<usize> {
    let mut vertices = vec![0; dim + 1];
    n -= 1;

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
```

**get_max_vertex** 优化（k=2特殊处理）:
```rust
fn get_max_vertex(index: i64, k: usize, n: usize, binomial: &BinomialCoeffTable) -> usize {
    if k == 2 {
        // 精确公式: C(n, 2) = n*(n-1)/2
        // 求解: n*(n-1)/2 = index
        let sqrt_arg = 2.0 * index as f64 + 0.25;
        return (sqrt_arg.sqrt().round()) as usize;
    }

    // 二分查找
    binary_search_predicate(n, |v| binomial.get(v, k) <= index)
}
```

### 3. 0维同调（连通分量）

**算法**: Kruskal 最小生成树 + Union-Find

```rust
fn compute_dim_0_pairs(edges: &[DiameterEntry], n: usize) -> Vec<(f32, f32)> {
    let mut dset = UnionFind::new(n);
    let mut pairs = Vec::new();

    for edge in edges.iter().rev() {  // 按直径降序
        let (i, j) = get_edge_vertices(edge.index);
        let u = dset.find(i);
        let v = dset.find(j);

        if u != v {
            // 连接两个分量
            let birth_vertex = dset.link_and_get_birth(u, v);
            let birth = dset.get_birth(birth_vertex);
            let death = edge.diameter;

            if death > birth {
                pairs.push((birth, death));
            }
        }
    }

    // 添加无穷长的连通分量
    for i in 0..n {
        if dset.find(i) == i {
            pairs.push((dset.get_birth(i), f32::INFINITY));
        }
    }

    pairs
}
```

**Union-Find** 扩展（支持 birth time）:
```rust
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
    birth: Vec<f32>,        // 分量的最早 birth 时间
    birth_idxs: Vec<usize>, // birth 顶点索引
}

impl UnionFind {
    fn link_and_get_birth(&mut self, u: usize, v: usize) -> usize {
        // Elder rule: 较年轻的分量（较大的 birth）先死
        if self.rank[u] < self.rank[v] {
            self.parent[u] = v;
            if self.birth[u] > self.birth[v] {
                self.birth_idxs[v]  // 返回较年轻的
            } else {
                self.birth_idxs[u]
            }
        } else {
            self.parent[v] = u;
            if self.rank[u] == self.rank[v] {
                self.rank[u] += 1;
            }
            // ...类似逻辑
        }
    }
}
```

**复杂度**: O(m α(n))，α 是 Ackermann 函数的逆

### 4. 高维同调（矩阵归约）

**核心思想**: 通过稀疏矩阵的列归约计算持久性

#### 4.1 边界矩阵（Boundary Matrix）

对于 k-simplex σ，其边界 ∂σ 是所有 (k-1)-face 的线性组合：

```
∂({v0, v1, v2}) = {v1, v2} - {v0, v2} + {v0, v1}
```

**边界枚举器**:
```rust
struct SimplexBoundaryEnumerator {
    simplex: DiameterEntry,
    dim: usize,
    idx_below: i64,
    idx_above: i64,
    k: usize,
    j: usize,
}

impl SimplexBoundaryEnumerator {
    fn next(&mut self, binomial: &BinomialCoeffTable) -> DiameterEntry {
        // 解码找到下一个顶点 j
        self.j = get_max_vertex(self.idx_below, self.k + 1, self.j, binomial);

        // 计算 face 的索引
        let face_index = self.idx_above - binomial.get(self.j, self.k + 1) + self.idx_below;

        // 计算符号（交替）
        let sign = if self.k & 1 == 1 { -1 } else { 1 };

        // 更新状态
        self.idx_below -= binomial.get(self.j, self.k + 1);
        self.idx_above += binomial.get(self.j, self.k);
        self.k -= 1;

        DiameterEntry::new(compute_diameter(face_index, self.dim - 1), face_index, sign)
    }
}
```

#### 4.2 Coboundary（上边界）

对于 k-simplex σ，其 coboundary δσ 是所有包含 σ 的 (k+1)-simplex：

```rust
struct SimplexCoboundaryEnumerator {
    simplex: DiameterEntry,
    dim: usize,
    vertices: Vec<usize>,  // simplex 的顶点
    next_vertex: usize,    // 候选顶点
}

impl SimplexCoboundaryEnumerator {
    fn next(&mut self, dist: &DistanceMatrix, binomial: &BinomialCoeffTable) -> DiameterEntry {
        // 找下一个不在 simplex 中的顶点
        while self.vertices.contains(&self.next_vertex) {
            self.next_vertex += 1;
        }

        // 计算 cofacet（添加 next_vertex 后的 simplex）
        let mut cofacet_vertices = self.vertices.clone();
        cofacet_vertices.push(self.next_vertex);
        cofacet_vertices.sort();

        // 计算直径（所有边的最大距离）
        let diameter = compute_diameter(&cofacet_vertices, dist);

        // 编码为索引
        let index = encode_simplex(&cofacet_vertices, binomial);

        self.next_vertex += 1;

        DiameterEntry::new(diameter, index, compute_coefficient(...))
    }
}
```

#### 4.3 矩阵归约算法

**伪代码**:
```
对每个待归约的列 c：
    working_boundary = ∂c（计算边界）

    while working_boundary 非空：
        pivot = working_boundary 的最大元素（按索引）

        if pivot 在 pivot_table 中：
            // 消元：working_boundary += 归约矩阵的某列
            column_to_add = pivot_table[pivot]
            working_boundary += reduction_matrix[column_to_add]
        else：
            // 找到新的持久性对
            pivot_table[pivot] = c
            birth = c 的直径
            death = pivot 的直径
            记录持久性对 (birth, death)
            break
```

**Rust 实现骨架**:
```rust
fn compute_pairs(
    columns_to_reduce: &[DiameterIndex],
    dim: usize,
) -> Vec<(f32, f32)> {
    let mut pivot_table = HashMap::new();
    let mut reduction_matrix = CompressedSparseMatrix::new(columns_to_reduce.len());
    let mut pairs = Vec::new();

    for (idx, &column_to_reduce) in columns_to_reduce.iter().enumerate() {
        let mut working_boundary = PriorityQueue::new();

        // 初始化边界
        let coboundary = compute_coboundary(column_to_reduce, dim);
        for entry in coboundary {
            working_boundary.push(entry);
        }

        // 归约循环
        loop {
            let pivot = pop_pivot(&mut working_boundary);

            if pivot.index == -1 {
                // 列已归约为0，无持久性对
                break;
            }

            if let Some(&column_to_add) = pivot_table.get(&pivot.index) {
                // 消元
                add_column(&mut working_boundary, &reduction_matrix, column_to_add, dim);
            } else {
                // 新的持久性对
                pivot_table.insert(pivot.index, idx);

                let birth = column_to_reduce.diameter;
                let death = pivot.diameter;

                if death != f32::INFINITY {
                    pairs.push((birth, death));
                }

                // 保存归约列
                reduction_matrix.set_column(idx, working_boundary);
                break;
            }
        }
    }

    pairs
}
```

**pop_pivot** 实现（Z/2Z 情况）:
```rust
fn pop_pivot(column: &mut PriorityQueue<DiameterEntry>) -> DiameterEntry {
    let mut pivot = DiameterEntry::invalid();

    while !column.is_empty() {
        pivot = column.pop().unwrap();

        if column.is_empty() || column.peek().unwrap().index != pivot.index {
            // 找到 pivot（出现奇数次）
            return pivot;
        }

        // 出现偶数次，抵消
        column.pop();
    }

    DiameterEntry::invalid()  // 列为空
}
```

**复杂度**:
- 最坏: O(n^3)（n = 简单复形数量）
- 实际: 通常接近 O(n log n)（稀疏性）

### 5. Apparent Pairs 优化

**思想**: 某些持久性对可以直接识别，无需矩阵归约

**定义**: 如果 k-simplex σ 和 (k+1)-simplex τ 满足：
1. σ 是 τ 的一个面
2. σ 和 τ 的直径相同
3. τ 的所有其他面的直径都更大

则 (σ, τ) 是一个 **apparent pair**（明显对）。

**检测**:
```rust
fn is_apparent_pair(sigma: &DiameterEntry, dim: usize) -> bool {
    // 找 sigma 的 coboundary 中与其直径相同的 cofacet
    let cofacet = find_zero_diameter_cofacet(sigma, dim);

    if cofacet.is_valid() {
        // 检查 cofacet 的所有 facet 中，只有 sigma 与其直径相同
        let facet = find_zero_diameter_facet(&cofacet, dim + 1);
        return facet.index == sigma.index;
    }

    false
}
```

**优势**:
- 避免矩阵归约（O(1) vs O(n²)）
- 大幅减少待归约的列数
- 典型场景加速 2-10倍

---

## 并行化策略

### 1. 维度间独立

**观察**: 不同维度的同调计算可以并行

```rust
fn compute_barcodes_parallel(edges: Vec<DiameterEntry>, max_dim: usize) -> Vec<Vec<(f32, f32)>> {
    let mut dgms = vec![Vec::new(); max_dim + 1];

    // H0 必须先计算（生成 H1 的候选列）
    dgms[0] = compute_dim_0_pairs(&edges);

    // H1, H2, ..., Hk 可以并行
    let handles: Vec<_> = (1..=max_dim).map(|dim| {
        thread::spawn(move || {
            compute_dim_pairs(dim)
        })
    }).collect();

    for (dim, handle) in handles.into_iter().enumerate() {
        dgms[dim + 1] = handle.join().unwrap();
    }

    dgms
}
```

### 2. 同维度内并行

**挑战**: 列归约存在依赖关系（pivot 冲突）

**解决方案**: Lock-free 归约（Morozov & Nigmetov, 2020）

**核心思想**:
- 使用原子操作管理 pivot_table
- 允许多个线程同时归约不同的列
- CAS（Compare-And-Swap）解决竞争

**伪代码**:
```rust
fn compute_pairs_parallel(
    columns_to_reduce: &[DiameterIndex],
    dim: usize,
    num_threads: usize,
) -> Vec<(f32, f32)> {
    let pivot_table = Arc::new(ConcurrentHashMap::new());
    let reduction_matrix = Arc::new(CompressedSparseMatrix::new(...));
    let pairs = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..num_threads).map(|t| {
        let pivot_table = Arc::clone(&pivot_table);
        let reduction_matrix = Arc::clone(&reduction_matrix);
        let pairs = Arc::clone(&pairs);

        thread::spawn(move || {
            for idx in (t..columns_to_reduce.len()).step_by(num_threads) {
                reduce_column_lockfree(
                    idx,
                    columns_to_reduce,
                    &pivot_table,
                    &reduction_matrix,
                    &pairs,
                    dim,
                );
            }
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(pairs).unwrap().into_inner().unwrap()
}

fn reduce_column_lockfree(
    idx: usize,
    columns_to_reduce: &[DiameterIndex],
    pivot_table: &ConcurrentHashMap<i64, usize>,
    reduction_matrix: &CompressedSparseMatrix,
    pairs: &Mutex<Vec<(f32, f32)>>,
    dim: usize,
) {
    loop {
        let mut working_boundary = compute_coboundary(...);

        loop {
            let pivot = pop_pivot(&mut working_boundary);

            if pivot.index == -1 {
                return;  // 列归约完成，无持久性对
            }

            match pivot_table.get(&pivot.index) {
                Some(column_to_add) => {
                    // 读到其他线程的结果，继续归约
                    add_column(&mut working_boundary, reduction_matrix, *column_to_add, dim);
                }
                None => {
                    // 尝试原子插入
                    if pivot_table.insert_if_absent(pivot.index, idx) {
                        // 成功插入，找到新的持久性对
                        let birth = columns_to_reduce[idx].diameter;
                        let death = pivot.diameter;
                        pairs.lock().unwrap().push((birth, death));
                        return;
                    } else {
                        // 插入失败，其他线程已占用此 pivot
                        // 重新读取并继续归约
                        continue;
                    }
                }
            }
        }
    }
}
```

**关键点**:
1. **ConcurrentHashMap**: 支持无锁并发插入/查询
2. **原子操作**: `insert_if_absent` 使用 CAS
3. **重试机制**: pivot 冲突时重新读取并继续
4. **无死锁**: 所有操作最终收敛

### 3. Rayon 并行化（Rust 推荐）

```rust
use rayon::prelude::*;

fn compute_pairs_rayon(
    columns_to_reduce: &[DiameterIndex],
    dim: usize,
) -> Vec<(f32, f32)> {
    let pivot_table = DashMap::new();  // 并发 HashMap
    let reduction_matrix = ConcurrentSparseMatrix::new(...);

    columns_to_reduce
        .par_iter()
        .enumerate()
        .filter_map(|(idx, &column_to_reduce)| {
            reduce_column_lockfree(
                idx,
                column_to_reduce,
                &pivot_table,
                &reduction_matrix,
                dim,
            )
        })
        .collect()
}
```

---

## Rust 实现路线图

### 迭代 1: 二项式系数表

**文件**: `src/ripser/core/binomial.rs`

**实现**:
```rust
pub struct BinomialCoeffTable {
    table: Vec<Vec<i64>>,
}

impl BinomialCoeffTable {
    pub fn new(n: usize, k: usize) -> Self;
    pub fn get(&self, n: usize, k: usize) -> i64;
}
```

**测试**:
- 与 Python `scipy.special.comb` 对比
- 边界条件（n=0, k=0, k>n）
- 溢出检测

### 迭代 2: 距离矩阵

**文件**: `src/ripser/core/distance.rs`

**实现**:
- `CompressedDistanceMatrix<LOWER_TRIANGULAR>`
- `CompressedDistanceMatrix<UPPER_TRIANGULAR>`
- `SparseDistanceMatrix`（COO 输入）

**测试**:
- 欧氏距离计算与 `scipy.spatial.distance.pdist` 对比
- 索引正确性
- 稀疏 vs 稠密性能

### 迭代 3-4: 简单复形

**文件**: `src/ripser/core/simplex.rs`

**实现**:
- `get_edge_index(i, j, binomial)`
- `get_simplex_vertices(index, dim, n, binomial)`
- `get_edges(dist, threshold, binomial)`
- `SimplexBoundaryEnumerator`
- `SimplexCoboundaryEnumerator`

**测试**:
- 编解码可逆性
- 边枚举完整性
- 与 C++ ripser 索引对比

### 迭代 5: 上同调计算

**文件**: `src/ripser/core/cohomology.rs`

**实现**:
- `UnionFind`（支持 birth time）
- `compute_dim_0_pairs(edges, n)`
- `CompressedSparseMatrix`
- `compute_pairs(columns_to_reduce, dim)`
- `pop_pivot(column)`

**测试**:
- 简单点云（圆形、球面）
- 与 giotto-ph 输出逐点对比
- 边界情况

### 迭代 6: 端到端集成

**文件**: `src/ripser/core/barcode.rs`

**实现**:
- `ripser(points, maxdim, thresh, coeff) -> RipserResults`
- 完整流程整合

**测试**:
- 1D 时间序列（你的使用场景）
- 持久熵计算验证

### 迭代 7: 并行化

**文件**: `src/ripser/parallel/`

**实现**:
- Rayon 并行边枚举
- Lock-free 矩阵归约
- 并发 HashMap（DashMap）

**测试**:
- 单线程 vs 多线程数值一致性
- 性能基准（加速比）

### 迭代 8: 高级优化

**实现**:
- Apparent pairs 优化
- Edge collapse
- Weighted filtration

---

## 参考文献

1. **Ulrich Bauer** (2021). "Ripser: efficient computation of Vietoris–Rips persistence barcodes." *Journal of Applied and Computational Topology*, 5, 391–423.

2. **Dmitriy Morozov & Arnur Nigmetov** (2020). "Towards Lockfree Persistent Homology." *SPAA '20*, 555–557.

3. **Edelsbrunner & Harer** (2010). *Computational Topology: An Introduction.* American Mathematical Society.

4. **giotto-ph** (2021). Python implementation and parallelization. https://github.com/giotto-ai/giotto-ph

---

## 附录：关键公式

### 二项式系数

```
C(n, k) = n! / (k! * (n-k)!)

递推: C(n, k) = C(n-1, k-1) + C(n-1, k)
```

### Simplex 索引编码

```
index(v0, v1, ..., vk) = Σ C(vi, i+1)  for i = 0..k
```

### 直径计算

```
diameter(σ) = max{ dist(vi, vj) | vi, vj ∈ σ }
```

### 持久性

```
persistence(σ) = death(σ) - birth(σ)
```

### 持久熵

```
L = { lifetime_i } = { death_i - birth_i }
P = L / sum(L)  (归一化为概率分布)
entropy = -Σ P_i * log2(P_i)
```

---

**下一步**: 开始迭代 1 - 实现二项式系数表
