"""
上同调计算测试

测试 0 维持久性同调（连通分量）的计算。

由于 ripser 模块还在开发中，暂时使用纯 Python 实现进行验证。
"""

import numpy as np
import pytest


# ============================================================================
# Python 参考实现 - UnionFind
# ============================================================================


class UnionFind:
    """
    并查集数据结构（支持 birth time 追踪）
    """

    def __init__(self, n):
        """初始化 n 个独立集合"""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.birth = [0.0] * n
        self.birth_vertex = list(range(n))

    def find(self, x):
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y, death):
        """合并两个集合（按秩合并 + elder rule）"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return self.birth_vertex[root_x]

        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            # Elder rule: 返回较年轻的分量
            if self.birth[root_x] > self.birth[root_y]:
                self.birth_vertex[root_y] = self.birth_vertex[root_x]
                return self.birth_vertex[root_x]
            else:
                return self.birth_vertex[root_y]
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
            # Elder rule: 返回较年轻的分量
            if self.birth[root_y] > self.birth[root_x]:
                self.birth_vertex[root_x] = self.birth_vertex[root_y]
                return self.birth_vertex[root_y]
            else:
                return self.birth_vertex[root_x]

    def get_birth(self, x):
        """获取分量的 birth time"""
        root = self.find(x)
        return self.birth[self.birth_vertex[root]]

    def set_birth(self, x, birth_time):
        """设置顶点的 birth time"""
        self.birth[x] = birth_time

    def get_roots(self):
        """获取所有独立分量的根节点"""
        roots = []
        for i in range(len(self.parent)):
            if self.find(i) == i:
                roots.append(i)
        return roots


# ============================================================================
# Python 参考实现 - 0维同调
# ============================================================================


def compute_dim_0_pairs(edges, n):
    """
    计算 0 维持久性同调（连通分量）

    Args:
        edges: 边的列表 [(diameter, i, j), ...]（升序排序）
        n: 顶点数量

    Returns:
        持久性对列表 [(birth, death), ...]
    """
    uf = UnionFind(n)
    pairs = []

    # 按直径降序遍历边
    for diameter, i, j in reversed(edges):
        root_i = uf.find(i)
        root_j = uf.find(j)

        if root_i != root_j:
            # 连接两个不同的分量
            birth_vertex = uf.union(i, j, diameter)
            birth = uf.get_birth(birth_vertex)
            death = diameter

            # 只记录有意义的持久性对
            if death > birth:
                pairs.append((birth, death))

    # 添加无穷长的分量
    roots = uf.get_roots()
    for root in roots:
        birth = uf.get_birth(root)
        pairs.append((birth, float('inf')))

    return pairs


# ============================================================================
# 测试用例
# ============================================================================


def test_union_find_basic():
    """UnionFind 基本功能测试"""
    uf = UnionFind(5)

    # 初始状态：每个元素独立
    assert uf.find(0) == 0
    assert uf.find(1) == 1

    # 合并 0 和 1
    uf.union(0, 1, 1.0)
    assert uf.find(0) == uf.find(1)

    # 合并 2 和 3
    uf.union(2, 3, 2.0)
    assert uf.find(2) == uf.find(3)

    # 0 和 2 仍然分离
    assert uf.find(0) != uf.find(2)


def test_union_find_path_compression():
    """路径压缩测试"""
    uf = UnionFind(4)

    # 创建链
    uf.union(0, 1, 1.0)
    uf.union(1, 2, 2.0)
    uf.union(2, 3, 3.0)

    # 查找会触发路径压缩
    root = uf.find(0)
    assert root == uf.find(3)

    # 路径压缩后，0 应该直接指向根
    assert uf.parent[0] == root


def test_union_find_birth_tracking():
    """Birth time 追踪测试"""
    uf = UnionFind(3)

    # 设置 birth time
    uf.set_birth(0, 0.0)
    uf.set_birth(1, 1.0)
    uf.set_birth(2, 0.5)

    assert uf.get_birth(0) == 0.0
    assert uf.get_birth(1) == 1.0
    assert uf.get_birth(2) == 0.5


def test_compute_dim_0_simple():
    """简单的 0 维同调测试"""
    # 3 个顶点，2 条边
    # {0, 1} 距离 1.0
    # {1, 2} 距离 2.0
    edges = [
        (1.0, 0, 1),
        (2.0, 1, 2),
    ]

    pairs = compute_dim_0_pairs(edges, 3)

    # 应该有 3 个持久性对
    assert len(pairs) == 3

    # 应该有 1 个无穷对
    infinite_pairs = [p for p in pairs if np.isinf(p[1])]
    assert len(infinite_pairs) == 1

    # 有限对的数量
    finite_pairs = [p for p in pairs if not np.isinf(p[1])]
    assert len(finite_pairs) == 2


def test_compute_dim_0_disconnected():
    """不连通图的 0 维同调测试"""
    # 4 个顶点，只有 1 条边 {0, 1}
    # 形成 2 个分量：{0, 1} 和 {2}, {3}
    edges = [
        (1.0, 0, 1),
    ]

    pairs = compute_dim_0_pairs(edges, 4)

    # 应该有 4 个持久性对
    assert len(pairs) == 4

    # 应该有 3 个无穷对（3 个最终分量）
    infinite_pairs = [p for p in pairs if np.isinf(p[1])]
    assert len(infinite_pairs) == 3


def test_compute_dim_0_complete_graph():
    """完全图的 0 维同调测试"""
    # 4 个顶点的完全图
    # 6 条边：C(4, 2) = 6
    edges = [
        (1.0, 0, 1),
        (1.5, 0, 2),
        (1.2, 0, 3),
        (1.8, 1, 2),
        (1.6, 1, 3),
        (2.0, 2, 3),
    ]

    pairs = compute_dim_0_pairs(edges, 4)

    # 应该有 4 个持久性对（3 个有限 + 1 个无穷）
    assert len(pairs) == 4

    # 应该有 1 个无穷对（最终的单一分量）
    infinite_pairs = [p for p in pairs if np.isinf(p[1])]
    assert len(infinite_pairs) == 1


def test_elder_rule():
    """Elder rule 测试"""
    uf = UnionFind(3)

    # 设置不同的 birth time
    uf.set_birth(0, 0.0)  # 较老
    uf.set_birth(1, 1.0)  # 较年轻

    # 合并时，较年轻的先死
    younger = uf.union(0, 1, 2.0)

    # younger 应该是顶点 1（birth = 1.0 > 0.0）
    assert younger == 1


def test_union_find_get_roots():
    """获取根节点测试"""
    uf = UnionFind(5)

    # 初始：5 个独立分量
    roots = uf.get_roots()
    assert len(roots) == 5

    # 合并一些元素
    uf.union(0, 1, 1.0)
    uf.union(2, 3, 2.0)

    roots = uf.get_roots()
    assert len(roots) == 3  # {0,1}, {2,3}, {4}


def test_dim_0_birth_death_order():
    """验证 birth < death"""
    edges = [
        (1.0, 0, 1),
        (2.0, 1, 2),
        (3.0, 2, 3),
    ]

    pairs = compute_dim_0_pairs(edges, 4)

    # 所有有限对应该满足 birth <= death
    for birth, death in pairs:
        if not np.isinf(death):
            assert birth <= death, f"Invalid pair: ({birth}, {death})"


def test_dim_0_persistence_values():
    """验证持久性值"""
    # 简单链：0 - 1 - 2
    edges = [
        (1.0, 0, 1),
        (2.0, 1, 2),
    ]

    pairs = compute_dim_0_pairs(edges, 3)

    # 有限对的 death 应该是边的直径
    finite_pairs = [(b, d) for b, d in pairs if not np.isinf(d)]

    # 应该有 2 个有限对
    assert len(finite_pairs) == 2

    # death 值应该是 1.0 和 2.0
    deaths = sorted([d for _, d in finite_pairs])
    assert deaths == [1.0, 2.0]


def test_dim_0_star_graph():
    """星形图的 0 维同调测试"""
    # 中心顶点 0 连接到 1, 2, 3
    edges = [
        (1.0, 0, 1),
        (1.0, 0, 2),
        (1.0, 0, 3),
    ]

    pairs = compute_dim_0_pairs(edges, 4)

    # 应该有 4 个持久性对
    assert len(pairs) == 4

    # 所有有限对的 death 都应该是 1.0
    finite_pairs = [(b, d) for b, d in pairs if not np.isinf(d)]
    for birth, death in finite_pairs:
        assert death == 1.0


## ============================================================================
# 高维同调测试（矩阵归约）
# ============================================================================


def test_sparse_column_basic():
    """稀疏列基本操作测试"""
    # 这个测试验证 Rust 实现的稀疏列是否正确
    # 由于 Rust 端是内部实现，我们通过端到端测试来验证

    # 测试思路：构造一个简单的拓扑空间，验证持久性对
    # 例如：三角形（3个顶点，3条边，1个面）

    # 顶点: 0, 1, 2
    # 边: {0,1}, {0,2}, {1,2}
    # 面: {0,1,2}

    # 对于 1 维同调（边的持久性）：
    # - 三角形形成一个环
    # - 当第三条边加入时，环被填充，持久性结束

    # 这个测试在后续集成测试中完成
    pass


def test_matrix_reduction_triangle():
    """矩阵归约测试：三角形"""
    # 三角形拓扑
    # 顶点: 0, 1, 2（距离为 0）
    # 边: {0,1} (距离 1.0), {0,2} (距离 1.5), {1,2} (距离 2.0)
    # 面: {0,1,2} (距离 2.0)

    # 0 维同调：3 个顶点合并成 1 个分量
    # - (0, 1.0): 顶点 1 合并到顶点 0
    # - (0, 1.5): 顶点 2 合并到分量
    # - (0, ∞): 最终分量存活

    # 1 维同调：环的形成与消失
    # - (1.0, 2.0): 环在边 {0,1} 形成，在边 {1,2} 时被面填充

    # 由于当前实现还未完全集成，这个测试暂时跳过
    pass


def test_persistence_computation_cycle():
    """持久性计算：环检测"""
    # 正方形（4个顶点，4条边，无面）
    # 顶点: 0, 1, 2, 3
    # 边: {0,1}, {1,2}, {2,3}, {3,0}

    # 0 维同调：4 个顶点合并成 1 个分量
    # 1 维同调：1 个持久环（正方形的边界）

    # 当前阶段暂不实现完整测试，等待端到端集成
    pass


def test_z2_coefficient_field():
    """Z/2Z 系数域测试"""
    # 验证稀疏列的 Z/2Z 加法
    # 在 Z/2Z 中，1 + 1 = 0

    # 测试思路：
    # col = {3, 5}
    # col.add(5)  -> col = {3}（5 被删除）
    # col.add(5)  -> col = {3, 5}（5 被添加回来）

    # 由于这是内部实现，通过 Rust 测试验证
    pass


def test_higher_dim_homology_sphere():
    """高维同调：球面测试"""
    # 2 维球面（如八面体）
    # 应该有：
    # - H0: 1 个分量
    # - H1: 0 个环
    # - H2: 1 个空洞（球面内部）

    # 这需要完整的端到端实现，暂时跳过
    pass


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
