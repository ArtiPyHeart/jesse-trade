//! FTI (Frequency Tunable Indicator) 核心算法实现
//!
//! 对应 Python fti.py 的实现

use ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

/// 计算特定周期的滤波器系数
///
/// 对应 Python 中的 _find_coefs_numba
pub fn find_coefs(period: usize, half_length: usize) -> Array1<f64> {
    // 系数设置
    let d = [0.35577019, 0.2436983, 0.07211497, 0.00630165];
    let mut coefs = Array1::<f64>::zeros(half_length + 1);

    // 计算中心系数
    let mut fact = 2.0 / period as f64;
    coefs[0] = fact;

    // 计算其他系数
    fact *= PI;
    for i in 1..=half_length {
        coefs[i] = (i as f64 * fact).sin() / (i as f64 * PI);
    }

    // 调整末端点
    coefs[half_length] *= 0.5;

    // 应用加权窗口并归一化
    let mut sumg = coefs[0];
    for i in 1..=half_length {
        let mut sum_val = d[0];
        let fact_i = i as f64 * PI / half_length as f64;
        for j in 1..4 {
            sum_val += 2.0 * d[j] * (j as f64 * fact_i).cos();
        }
        coefs[i] *= sum_val;
        sumg += 2.0 * coefs[i];
    }

    // 归一化系数
    if sumg != 0.0 {
        coefs /= sumg;
    }

    coefs
}

/// 使用最小二乘线外推数据
///
/// 对应 Python 中的 _extrapolate_data_numba
pub fn extrapolate_data(y: &mut Array1<f64>, lookback: usize, half_length: usize) {
    // 计算最近half_length+1个数据点的均值
    let xmean = -0.5 * half_length as f64;
    let mut ymean = 0.0;
    for i in (lookback - half_length - 1)..lookback {
        ymean += y[i];
    }
    ymean /= (half_length + 1) as f64;

    // 计算最小二乘线的斜率
    let mut xsq = 0.0;
    let mut xy = 0.0;
    for i in 0..=half_length {
        let xdiff = -(i as f64) - xmean;
        let ydiff = y[lookback - 1 - i] - ymean;
        xsq += xdiff * xdiff;
        xy += xdiff * ydiff;
    }

    let slope = if xsq != 0.0 { xy / xsq } else { 0.0 };

    // 扩展数据
    for i in 0..half_length {
        y[lookback + i] = ((i + 1) as f64 - xmean) * slope + ymean;
    }
}

/// 应用滤波器结果
#[derive(Debug)]
pub struct FilterResult {
    pub filtered_value: f64,
    pub longest_leg: f64,
    pub n_legs: usize,
    pub diff_work: Array1<f64>,
    pub leg_work: Array1<f64>,
}

/// 应用滤波器并收集移动腿
///
/// 对应 Python 中的 _apply_filter_numba
pub fn apply_filter(
    y: &ArrayView1<f64>,
    coefs: &ArrayView1<f64>,
    half_length: usize,
    lookback: usize,
) -> FilterResult {
    let mut diff_work = Array1::<f64>::zeros(lookback);
    let mut leg_work = Array1::<f64>::zeros(lookback);

    // 初始化变量
    let mut extreme_type = 0; // 未定义。1=高点; -1=低点
    let mut extreme_value = 0.0;
    let mut n_legs = 0;
    let mut longest_leg = 0.0;
    let mut prior = 0.0;
    let mut filtered_value = 0.0;

    // 对数据块中的每个点应用滤波器
    for iy in half_length..lookback {
        // 应用卷积滤波器
        let mut sum_val = coefs[0] * y[iy]; // 中心点
        for i in 1..=half_length {
            sum_val += coefs[i] * (y[iy + i] + y[iy - i]); // 对称滤波
        }

        // 如果这是当前数据点的滤波值，保存它
        if iy == lookback - 1 {
            filtered_value = sum_val;
        }

        // 保存实际值与滤波值之间的差异，用于宽度计算
        diff_work[iy - half_length] = (y[iy] - sum_val).abs();

        // 收集移动腿
        if iy == half_length {
            // 第一个点
            extreme_type = 0;
            extreme_value = sum_val;
            n_legs = 0;
            longest_leg = 0.0;
        } else if extreme_type == 0 {
            // 等待第一个滤波价格变化
            if sum_val > extreme_value {
                extreme_type = -1; // 第一个点是低点
            } else if sum_val < extreme_value {
                extreme_type = 1; // 第一个点是高点
            }
        } else if iy == lookback - 1 {
            // 最后一点，视为转折点
            if extreme_type != 0 {
                let leg_length = (extreme_value - sum_val).abs();
                leg_work[n_legs] = leg_length;
                n_legs += 1;
                if leg_length > longest_leg {
                    longest_leg = leg_length;
                }
            }
        } else {
            // 内部前进
            if extreme_type == 1 && sum_val > prior {
                // 下降后转为上升
                let leg_length = extreme_value - prior;
                leg_work[n_legs] = leg_length;
                n_legs += 1;
                if leg_length > longest_leg {
                    longest_leg = leg_length;
                }
                extreme_type = -1;
                extreme_value = prior;
            } else if extreme_type == -1 && sum_val < prior {
                // 上升后转为下降
                let leg_length = prior - extreme_value;
                leg_work[n_legs] = leg_length;
                n_legs += 1;
                if leg_length > longest_leg {
                    longest_leg = leg_length;
                }
                extreme_type = 1;
                extreme_value = prior;
            }
        }

        prior = sum_val;
    }

    FilterResult {
        filtered_value,
        longest_leg,
        n_legs,
        diff_work,
        leg_work,
    }
}

/// 计算通道宽度
///
/// 对应 Python 中的 _calculate_width_numba
pub fn calculate_width(
    diff_work: &ArrayView1<f64>,
    lookback: usize,
    half_length: usize,
    beta: f64,
) -> f64 {
    // 创建副本进行排序以避免修改原始数组
    let n = lookback - half_length;
    let mut sorted_diffs = diff_work.slice(s![..n]).to_owned();
    sorted_diffs.as_slice_mut().unwrap().sort_by(|a, b| a.partial_cmp(b).unwrap());

    let i = ((beta * n as f64) as usize).saturating_sub(1);

    // 返回通道宽度
    sorted_diffs[i]
}

/// 计算FTI值
///
/// 对应 Python 中的 _calculate_fti_numba
pub fn calculate_fti(
    leg_work: &ArrayView1<f64>,
    width: f64,
    n_legs: usize,
    longest_leg: f64,
    noise_cut: f64,
) -> f64 {
    // 计算噪声水平
    let noise_level = noise_cut * longest_leg;

    // 计算所有大于噪声水平的腿的平均值
    let mut sum_val = 0.0;
    let mut n = 0;
    for i in 0..n_legs {
        if leg_work[i] > noise_level {
            sum_val += leg_work[i];
            n += 1;
        }
    }

    // 计算非噪声腿的平均移动
    if n > 0 {
        let mean_move = sum_val / n as f64;
        mean_move / (width + 1.0e-5)
    } else {
        0.0
    }
}

/// 排序FTI局部最大值并保存排序后的索引
///
/// 对应 Python 中的 _sort_local_maxima_numba
pub fn sort_local_maxima(fti_values: &ArrayView1<f64>) -> Array1<usize> {
    let num_periods = fti_values.len();
    let mut sorted_indices = Array1::<usize>::zeros(num_periods);
    let mut sort_work = Array1::<f64>::zeros(num_periods);

    // 找到局部最大值（包括两个端点）
    let mut n = 0;
    for i in 0..num_periods {
        if i == 0
            || i == num_periods - 1
            || (fti_values[i] >= fti_values[i - 1] && fti_values[i] >= fti_values[i + 1])
        {
            sort_work[n] = -fti_values[i]; // 要降序排列FTI，但排序是升序
            sorted_indices[n] = i;
            n += 1;
        }
    }

    // 对局部最大值进行排序（简单冒泡排序）
    for i in 0..n {
        for j in (i + 1)..n {
            if sort_work[i] > sort_work[j] {
                // 交换值和索引
                let temp_val = sort_work[i];
                let temp_idx = sorted_indices[i];
                sort_work[i] = sort_work[j];
                sorted_indices[i] = sorted_indices[j];
                sort_work[j] = temp_val;
                sorted_indices[j] = temp_idx;
            }
        }
    }

    sorted_indices
}

/// FTI处理结果
#[derive(Debug, Clone)]
pub struct FTIResult {
    pub fti: f64,
    pub filtered_value: f64,
    pub width: f64,
    pub best_period: f64,
}

/// FTI计算器
pub struct FTI {
    pub use_log: bool,
    pub min_period: usize,
    pub max_period: usize,
    pub half_length: usize,
    pub lookback: usize,
    pub beta: f64,
    pub noise_cut: f64,

    // 预分配的数组
    y: Array1<f64>,
    coefs: Array2<f64>,
    filtered: Array1<f64>,
    width: Array1<f64>,
    fti_values: Array1<f64>,
}

impl FTI {
    /// 创建新的FTI计算器
    pub fn new(
        use_log: bool,
        min_period: usize,
        max_period: usize,
        half_length: usize,
        lookback: usize,
        beta: f64,
        noise_cut: f64,
    ) -> Result<Self, String> {
        // 检查参数有效性
        if max_period < min_period || min_period < 2 {
            return Err("max_period必须大于min_period且min_period至少为2".to_string());
        }
        if 2 * half_length < max_period {
            return Err("2*half_length必须大于max_period".to_string());
        }
        if lookback < half_length + 2 {
            return Err("lookback必须比half_length至少大2".to_string());
        }

        let num_periods = max_period - min_period + 1;

        // 初始化数组
        let y = Array1::<f64>::zeros(lookback + half_length);
        let mut coefs = Array2::<f64>::zeros((num_periods, half_length + 1));
        let filtered = Array1::<f64>::zeros(num_periods);
        let width = Array1::<f64>::zeros(num_periods);
        let fti_values = Array1::<f64>::zeros(num_periods);

        // 计算每个周期的滤波器系数
        for (i, period) in (min_period..=max_period).enumerate() {
            let period_coefs = find_coefs(period, half_length);
            coefs.row_mut(i).assign(&period_coefs);
        }

        Ok(FTI {
            use_log,
            min_period,
            max_period,
            half_length,
            lookback,
            beta,
            noise_cut,
            y,
            coefs,
            filtered,
            width,
            fti_values,
        })
    }

    /// 处理价格数据块并计算FTI指标
    ///
    /// data: 价格数据，最近的数据点在索引0
    pub fn process(&mut self, data: &ArrayView1<f64>) -> Result<FTIResult, String> {
        // 检查数据长度
        if data.len() < self.lookback {
            return Err(format!("数据长度必须至少为{}", self.lookback));
        }

        // 收集数据到本地数组，使其按时间顺序排列
        // 最近的案例将在索引lookback-1
        for i in 0..self.lookback {
            if self.use_log {
                self.y[self.lookback - 1 - i] = data[i].ln();
            } else {
                self.y[self.lookback - 1 - i] = data[i];
            }
        }

        // 拟合最小二乘线并扩展
        extrapolate_data(&mut self.y, self.lookback, self.half_length);

        // 处理每个周期
        for (period_idx, period) in (self.min_period..=self.max_period).enumerate() {
            self.process_period(period_idx, period);
        }

        // 排序FTI局部最大值并保存排序后的索引
        let sorted = sort_local_maxima(&self.fti_values.view());

        // 返回结果
        let best_idx = sorted[0];
        let best_period = self.min_period + best_idx;

        // 获取相应的指标值
        let (filtered_value, width_value) = if self.use_log {
            let fv = self.filtered[best_idx].exp();
            let wv = 0.5
                * ((self.filtered[best_idx] + self.width[best_idx]).exp()
                    - (self.filtered[best_idx] - self.width[best_idx]).exp());
            (fv, wv)
        } else {
            (self.filtered[best_idx], self.width[best_idx])
        };

        // 计算最终的FTI值，包括Gamma累积分布函数变换
        let fti_value = self.fti_values[best_idx];
        let fti_transformed = 100.0 * gammainc(2.0, fti_value / 3.0) - 50.0;

        Ok(FTIResult {
            fti: fti_transformed,
            filtered_value,
            width: width_value,
            best_period: best_period as f64,
        })
    }

    fn process_period(&mut self, period_idx: usize, _period: usize) {
        // 获取该周期的滤波器系数
        let coefs = self.coefs.row(period_idx);

        // 应用滤波器到数据块中的每个值
        let filter_result = apply_filter(&self.y.view(), &coefs, self.half_length, self.lookback);

        // 保存滤波值
        self.filtered[period_idx] = filter_result.filtered_value;

        // 计算通道宽度
        self.width[period_idx] = calculate_width(
            &filter_result.diff_work.view(),
            self.lookback,
            self.half_length,
            self.beta,
        );

        // 计算FTI值
        self.fti_values[period_idx] = calculate_fti(
            &filter_result.leg_work.view(),
            self.width[period_idx],
            filter_result.n_legs,
            filter_result.longest_leg,
            self.noise_cut,
        );
    }
}

/// Gamma累积分布函数的近似实现
///
/// 对应 scipy.special.gammainc
fn gammainc(a: f64, x: f64) -> f64 {
    // 使用级数展开近似
    // P(a,x) = gamma(a,x)/Gamma(a) = 1 - exp(-x) * sum(x^k / (a+k)!)

    if x <= 0.0 {
        return 0.0;
    }

    // 对于 a=2, 有简化形式: P(2,x) = 1 - (1+x)*exp(-x)
    if (a - 2.0).abs() < 1e-10 {
        return 1.0 - (1.0 + x) * (-x).exp();
    }

    // 一般情况使用级数展开
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    sum += term;

    for k in 1..100 {
        term *= x / (a + k as f64);
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }

    let result = (-x).exp() * x.powf(a) * sum / gamma(a);
    result.min(1.0).max(0.0)
}

/// Gamma函数近似
fn gamma(z: f64) -> f64 {
    // 对于整数参数，使用阶乘
    if (z - 2.0).abs() < 1e-10 {
        return 1.0; // Gamma(2) = 1! = 1
    }

    // Stirling近似
    let sqrt_2pi = (2.0 * PI).sqrt();
    sqrt_2pi * z.powf(z - 0.5) * (-z).exp()
}

// 需要导入 s! 宏来使用切片
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_coefs() {
        let coefs = find_coefs(10, 5);
        assert_eq!(coefs.len(), 6);

        // 系数和应该约为1（归一化后）
        let sum: f64 = coefs[0] + 2.0 * coefs.iter().skip(1).sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolate_data() {
        let mut y = Array1::<f64>::zeros(10);
        for i in 0..5 {
            y[i] = i as f64;
        }
        extrapolate_data(&mut y, 5, 2);

        // 外推的数据应该是合理的
        assert!(y[5].is_finite());
        assert!(y[6].is_finite());
    }

    #[test]
    fn test_fti_creation() {
        let fti = FTI::new(true, 5, 65, 35, 150, 0.95, 0.20);
        assert!(fti.is_ok());
    }

    #[test]
    fn test_fti_invalid_params() {
        let fti = FTI::new(true, 65, 5, 35, 150, 0.95, 0.20);
        assert!(fti.is_err());
    }
}
