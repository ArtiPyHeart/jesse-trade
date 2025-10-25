/*!
 * 核心数据类型定义
 *
 * 这个模块定义了 Ripser 算法中使用的所有核心数据类型。
 *
 * # 设计原则
 *
 * 1. **类型安全**: 使用强类型避免索引错误
 * 2. **内存效率**: 紧凑的数据表示
 * 3. **零成本抽象**: 编译时优化，运行时无开销
 *
 * # 核心类型
 *
 * - `Value`: 持久性值（浮点数）
 * - `Index`: 简单复形索引（整数）
 * - `Coefficient`: 系数域元素
 * - `DiameterEntry`: 直径-索引-系数三元组
 */

use std::fmt;

// ============================================================================
// 基础类型别名
// ============================================================================

/// 持久性值类型（filtration value）
///
/// 使用 f32 而非 f64 以节省内存，这与 C++ ripser 保持一致。
/// 对于大多数应用，f32 的精度已经足够。
pub type Value = f32;

/// 简单复形索引类型
///
/// 使用 i64 以支持大规模点云（最多约 10^9 个 simplex）。
/// 负值用于特殊标记（如 -1 表示无效索引）。
pub type Index = i64;

/// 系数域元素类型
///
/// 通常用于 Z/pZ（p 为素数）。默认情况下 p=2（Z/2Z）。
/// 使用 u16 支持素数最大到 65535。
pub type Coefficient = u16;

// ============================================================================
// DiameterEntry - 核心数据结构
// ============================================================================

/// 直径条目：(直径, 索引, 系数) 三元组
///
/// 这是 Ripser 中最核心的数据结构，用于表示过滤（filtration）中的简单复形。
///
/// # 字段
///
/// - `diameter`: 简单复形的直径（filtration value）
/// - `index`: 简单复形的组合编码索引
/// - `coefficient`: 系数域中的系数（用于上同调计算）
///
/// # 内存布局
///
/// 总大小：16 字节（4 + 8 + 2 + 2 对齐）
///
/// # 示例
///
/// ```ignore
/// let edge = DiameterEntry::new(1.5, 42, 1);
/// assert_eq!(edge.diameter, 1.5);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DiameterEntry {
    /// 简单复形的直径（filtration value）
    pub diameter: Value,

    /// 简单复形的组合编码索引
    pub index: Index,

    /// 系数域中的系数
    pub coefficient: Coefficient,
}

impl DiameterEntry {
    /// 创建新的直径条目
    #[inline]
    pub fn new(diameter: Value, index: Index, coefficient: Coefficient) -> Self {
        Self {
            diameter,
            index,
            coefficient,
        }
    }

    /// 创建无系数的直径条目（系数默认为 0）
    #[inline]
    pub fn new_without_coeff(diameter: Value, index: Index) -> Self {
        Self::new(diameter, index, 0)
    }

    /// 获取直径
    #[inline]
    pub fn get_diameter(&self) -> Value {
        self.diameter
    }

    /// 获取索引
    #[inline]
    pub fn get_index(&self) -> Index {
        self.index
    }

    /// 获取系数
    #[inline]
    pub fn get_coefficient(&self) -> Coefficient {
        self.coefficient
    }

    /// 设置系数
    #[inline]
    pub fn set_coefficient(&mut self, coeff: Coefficient) {
        self.coefficient = coeff;
    }
}

// 实现 PartialEq 和 Eq（基于索引比较）
impl PartialEq for DiameterEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for DiameterEntry {}

// 实现排序（按直径排序，直径相同则按索引）
impl PartialOrd for DiameterEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DiameterEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 先按直径排序
        match self.diameter.partial_cmp(&other.diameter) {
            Some(std::cmp::Ordering::Equal) => {
                // 直径相同，按索引排序
                self.index.cmp(&other.index)
            }
            Some(ord) => ord,
            // NaN 处理：NaN 视为最大
            None => {
                if self.diameter.is_nan() {
                    if other.diameter.is_nan() {
                        std::cmp::Ordering::Equal
                    } else {
                        std::cmp::Ordering::Greater
                    }
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    }
}

impl fmt::Display for DiameterEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiameterEntry(diameter={:.6}, index={}, coeff={})",
            self.diameter, self.index, self.coefficient
        )
    }
}

// ============================================================================
// DiameterIndex - 简化版本（不含系数）
// ============================================================================

/// 直径-索引对（不含系数）
///
/// 在某些场景下只需要直径和索引，不需要系数。
/// 使用这个更轻量的结构可以节省内存。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterIndex {
    pub diameter: Value,
    pub index: Index,
}

impl DiameterIndex {
    #[inline]
    pub fn new(diameter: Value, index: Index) -> Self {
        Self { diameter, index }
    }
}

impl From<DiameterEntry> for DiameterIndex {
    fn from(entry: DiameterEntry) -> Self {
        Self::new(entry.diameter, entry.index)
    }
}

// ============================================================================
// RipserResults - 结果数据结构
// ============================================================================

/// Ripser 计算结果
///
/// 包含各维度的持久性图（persistence diagrams）。
///
/// # 字段
///
/// - `dgms`: 持久性图列表，dgms[d] 是维度 d 的持久性图
///   - 每个持久性图是 Nx2 的矩阵：[[birth1, death1], [birth2, death2], ...]
///   - 无穷大的 death 用 f32::INFINITY 表示
///
/// # 示例
///
/// ```ignore
/// let results = RipserResults::new(2); // 计算 H0 和 H1
/// assert_eq!(results.dgms.len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct RipserResults {
    /// 各维度的持久性图
    /// dgms[d] = 维度 d 的持久性图（Nx2矩阵）
    pub dgms: Vec<Vec<(Value, Value)>>, // Vec of (birth, death) pairs
}

impl RipserResults {
    /// 创建新的结果结构（预分配指定维度）
    pub fn new(max_dim: usize) -> Self {
        Self {
            dgms: vec![Vec::new(); max_dim + 1],
        }
    }

    /// 添加持久性对到指定维度
    pub fn add_pair(&mut self, dim: usize, birth: Value, death: Value) {
        if dim < self.dgms.len() {
            self.dgms[dim].push((birth, death));
        }
    }

    /// 获取指定维度的持久性图
    pub fn get_diagram(&self, dim: usize) -> Option<&Vec<(Value, Value)>> {
        self.dgms.get(dim)
    }

    /// 获取总的持久性对数量
    pub fn total_pairs(&self) -> usize {
        self.dgms.iter().map(|d| d.len()).sum()
    }
}

impl fmt::Display for RipserResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RipserResults {{")?;
        for (dim, dgm) in self.dgms.iter().enumerate() {
            writeln!(f, "  H{}: {} pairs", dim, dgm.len())?;
        }
        write!(f, "}}")
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diameter_entry_creation() {
        let entry = DiameterEntry::new(1.5, 42, 1);
        assert_eq!(entry.diameter, 1.5);
        assert_eq!(entry.index, 42);
        assert_eq!(entry.coefficient, 1);
    }

    #[test]
    fn test_diameter_entry_ordering() {
        let e1 = DiameterEntry::new(1.0, 10, 1);
        let e2 = DiameterEntry::new(2.0, 20, 1);
        let e3 = DiameterEntry::new(1.0, 5, 1);

        assert!(e1 < e2); // 按直径排序
        assert!(e3 < e1); // 直径相同，按索引排序
    }

    #[test]
    fn test_diameter_entry_equality() {
        let e1 = DiameterEntry::new(1.0, 42, 1);
        let e2 = DiameterEntry::new(2.0, 42, 2);

        assert_eq!(e1, e2); // 相等性仅基于索引
    }

    #[test]
    fn test_ripser_results() {
        let mut results = RipserResults::new(2);

        // 添加 H0 的持久性对
        results.add_pair(0, 0.0, 1.5);
        results.add_pair(0, 0.0, 2.0);

        // 添加 H1 的持久性对
        results.add_pair(1, 1.0, 3.0);

        assert_eq!(results.dgms[0].len(), 2);
        assert_eq!(results.dgms[1].len(), 1);
        assert_eq!(results.total_pairs(), 3);
    }

    #[test]
    fn test_diameter_index_conversion() {
        let entry = DiameterEntry::new(1.5, 42, 1);
        let di: DiameterIndex = entry.into();

        assert_eq!(di.diameter, 1.5);
        assert_eq!(di.index, 42);
    }
}
