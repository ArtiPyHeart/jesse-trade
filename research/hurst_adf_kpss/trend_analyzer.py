import warnings
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
from scipy.stats import linregress
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


class TrendAnalyzer:
    """
    Jesse K线趋势的Hurst-ADF-KPSS三重检验系统

    用于判断价格序列的趋势特征：
    - Hurst指数: 衡量时间序列的长期记忆性和趋势持续性
    - ADF检验: 检验序列是否存在单位根（非平稳性）
    - KPSS检验: 检验序列是否以平稳为零假设（与ADF互补）
    """

    def __init__(self, candles: np.ndarray, close_column_idx: int = 2):
        """
        初始化趋势分析器

        Parameters:
        -----------
        candles: np.ndarray
            Jesse风格的K线数据，shape=(n, 6)
            列: [timestamp, open, close, high, low, volume]
        close_column_idx: int
            收盘价在K线数组中的列索引，默认为2
        """
        assert (
            candles.ndim == 2 and candles.shape[1] == 6
        ), "candles必须是shape=(n, 6)的Jesse风格numpy array"

        self.candles = candles
        self.close_prices = candles[:, close_column_idx]
        self.results = {}

    def calculate_hurst(
        self, series: np.ndarray, min_lag: int = 10, max_lag: int = 100
    ) -> float:
        """
        计算Hurst指数（R/S分析法）

        Parameters:
        -----------
        series: np.ndarray
            价格序列
        min_lag: int
            最小滞后期
        max_lag: int
            最大滞后期

        Returns:
        --------
        float: Hurst指数值
            - H < 0.5: 反持续性（均值回归）
            - H = 0.5: 随机游走（无趋势）
            - H > 0.5: 持续性（趋势）
            - H > 0.6: 强趋势潜力
        """
        # 确保max_lag不超过序列长度的一半
        max_lag = min(max_lag, len(series) // 2)

        # 检查是否有足够的数据点
        if max_lag < min_lag:
            return np.nan

        # 计算对数收益率
        log_returns = np.log(series[1:] / series[:-1])

        # 检查对数收益率是否有足够的数据点
        # 使用 < 而非 <=，因为 len == max_lag 时仍可计算一个窗口
        if len(log_returns) < max_lag:
            return np.nan

        lags = range(min_lag, max_lag + 1)
        valid_lags = []  # 记录有效的 lag，避免与 rs_values 长度不匹配
        rs_values = []

        for lag in lags:
            # 计算R/S统计量
            rs = []
            for i in range(len(log_returns) - lag + 1):
                segment = log_returns[i : i + lag]
                deviations = segment - segment.mean()
                cumulative = np.cumsum(deviations)
                range_ = cumulative.max() - cumulative.min()
                std_ = segment.std(ddof=1)  # 使用样本标准差 (N-1)
                if std_ != 0:  # 避免除以0
                    rs.append(range_ / std_)

            # 如果没有有效的R/S值，跳过这个滞后
            if not rs:
                continue
            valid_lags.append(lag)  # 同步记录有效的 lag
            rs_values.append(np.mean(rs))

        # 检查是否有足够的R/S值进行回归
        if len(rs_values) < 3:  # 至少需要3个点进行线性回归
            return np.nan

        # 线性回归计算Hurst指数
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                slope, _, _, _, _ = linregress(np.log(valid_lags), np.log(rs_values))
                return slope
        except Exception:
            return np.nan

    def run_adf_test(self, series: np.ndarray) -> Dict:
        """
        执行ADF（Augmented Dickey-Fuller）检验

        Parameters:
        -----------
        series: np.ndarray
            价格序列

        Returns:
        --------
        dict: ADF检验结果
            - statistic: ADF统计量
            - pvalue: p值（>0.05表示存在单位根，非平稳）
            - lags: 使用的滞后期数
            - critical_values: 临界值字典
        """
        # 处理可能的缺失值
        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 20:  # 数据点太少
            return {
                "statistic": np.nan,
                "pvalue": np.nan,
                "lags": 0,
                "critical_values": {},
            }

        try:
            result = adfuller(series_clean, autolag="AIC")
            return {
                "statistic": result[0],
                "pvalue": result[1],
                "lags": result[2],
                "critical_values": result[4],
            }
        except Exception:
            return {
                "statistic": np.nan,
                "pvalue": np.nan,
                "lags": 0,
                "critical_values": {},
            }

    def run_kpss_test(
        self, series: np.ndarray, regression: str = "ct", suppress_warnings: bool = True
    ) -> Dict:
        """
        执行KPSS（Kwiatkowski-Phillips-Schmidt-Shin）检验

        Parameters:
        -----------
        series: np.ndarray
            价格序列
        regression: str
            回归类型，'c'为常数，'ct'为常数加趋势
        suppress_warnings: bool
            是否抑制插值警告

        Returns:
        --------
        dict: KPSS检验结果
            - statistic: KPSS统计量
            - pvalue: p值（<0.05表示拒绝平稳假设）
            - critical_values: 临界值字典
            - warning: 可能的警告信息
        """
        # 处理可能的缺失值
        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 20:  # 数据点太少
            return {
                "statistic": np.nan,
                "pvalue": np.nan,
                "critical_values": {},
                "warning": "数据点不足",
            }

        warning_msg = None
        try:
            if suppress_warnings:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = kpss(series_clean, regression=regression)

                    for warning in w:
                        if isinstance(warning.message, InterpolationWarning):
                            warning_msg = str(warning.message)
                            # 简化警告信息
                            if "smaller" in warning_msg:
                                warning_msg = "p值可能更小"
                            elif "greater" in warning_msg:
                                warning_msg = "p值可能更大"
                            else:
                                warning_msg = "p值估计可能不准确"
            else:
                result = kpss(series_clean, regression=regression)

            return {
                "statistic": result[0],
                "pvalue": result[1],
                "critical_values": result[3],
                "warning": warning_msg,
            }
        except Exception as e:
            return {
                "statistic": np.nan,
                "pvalue": np.nan,
                "critical_values": {},
                "warning": str(e),
            }

    def _calculate_trend_score(
        self, hurst: float, adf_result: Dict, kpss_result: Dict
    ) -> int:
        """
        计算趋势适配性评分（0-5分）

        评分规则：
        - Hurst > 0.6: +2分
        - Hurst > 0.55: +1分
        - ADF p值 > 0.05（非平稳）: +1分
        - KPSS p值 < 0.05（拒绝平稳）: +1分
        - 三重确认（Hurst>0.55, ADF非平稳, KPSS拒绝平稳）: +1分
        """
        if np.isnan(hurst):
            return 0

        score = 0

        if hurst > 0.6:
            score += 2
        elif hurst > 0.55:
            score += 1

        if adf_result["pvalue"] > 0.05:
            score += 1

        if kpss_result["pvalue"] < 0.05:
            score += 1

        # 三重确认奖励
        if (
            hurst > 0.55
            and adf_result["pvalue"] > 0.05
            and kpss_result["pvalue"] < 0.05
        ):
            score += 1

        return min(score, 5)

    def _get_hypothesis_test_explanation(
        self, adf_result: Dict, kpss_result: Dict
    ) -> str:
        """
        生成假设检验的详细解释

        Returns:
        --------
        str: 包含原假设、检验结果和结论的解释
        """
        adf_p = adf_result["pvalue"]
        kpss_p = kpss_result["pvalue"]

        if np.isnan(adf_p) or np.isnan(kpss_p):
            return "数据不足，无法进行假设检验"

        # ADF 检验解释
        adf_reject = adf_p <= 0.05
        adf_conclusion = "拒绝原假设→序列平稳" if adf_reject else "无法拒绝原假设→序列非平稳"
        adf_explanation = f"ADF检验[H0:存在单位根(非平稳)]: p={adf_p:.4f}, {adf_conclusion}"

        # KPSS 检验解释
        kpss_reject = kpss_p < 0.05
        kpss_conclusion = "拒绝原假设→序列非平稳" if kpss_reject else "无法拒绝原假设→序列平稳"
        kpss_warning_str = ""
        if kpss_result.get("warning"):
            kpss_warning_str = f" (警告:{kpss_result['warning']})"
        kpss_explanation = f"KPSS检验[H0:序列平稳]: p={kpss_p:.4f}, {kpss_conclusion}{kpss_warning_str}"

        return f"{adf_explanation}; {kpss_explanation}"

    def _classify_trend_type(
        self, hurst: float, adf_result: Dict, kpss_result: Dict
    ) -> str:
        """
        基于三个指标分类趋势类型，并包含详细的假设检验解释
        """
        if np.isnan(hurst):
            return "无法确定（Hurst指数计算失败）"

        adf_p = adf_result["pvalue"]
        kpss_p = kpss_result["pvalue"]

        if np.isnan(adf_p) or np.isnan(kpss_p):
            return "数据不足"

        # 获取假设检验详细解释
        hypothesis_explanation = self._get_hypothesis_test_explanation(adf_result, kpss_result)

        # Hurst 解释
        if hurst > 0.6:
            hurst_desc = "强趋势持续性"
        elif hurst > 0.55:
            hurst_desc = "趋势持续性"
        elif hurst > 0.5:
            hurst_desc = "弱趋势持续性"
        elif hurst > 0.45:
            hurst_desc = "接近随机游走"
        else:
            hurst_desc = "反持续性(均值回归)"

        hurst_str = f"Hurst={hurst:.3f}({hurst_desc})"

        # 综合判断
        adf_nonstationary = adf_p > 0.05
        kpss_nonstationary = kpss_p < 0.05

        if hurst > 0.55 and adf_nonstationary and kpss_nonstationary:
            # 三重确认：强趋势
            return (f"【强趋势且非平稳-适合趋势策略】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  综合结论: 三重确认趋势特征，价格持续偏离均值，适合趋势跟踪")

        elif hurst > 0.55 and not adf_nonstationary and not kpss_nonstationary:
            # Hurst显示趋势，但两个检验都认为平稳
            return (f"【趋势但平稳-短期趋势可能】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  综合结论: 价格具有持续性但围绕均值波动，可能是短期趋势或波段行情")

        elif hurst < 0.5 and not adf_nonstationary and not kpss_nonstationary:
            # 震荡平稳
            return (f"【震荡平稳-不适合趋势策略】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  综合结论: 价格具有均值回归特性且序列平稳，适合震荡或均值回归策略")

        elif hurst > 0.55 and adf_nonstationary and not kpss_nonstationary:
            # 矛盾1: ADF说非平稳，KPSS说平稳，但Hurst显示趋势
            return (f"【矛盾-需进一步验证】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  ⚠️ 矛盾分析: ADF认为非平稳但KPSS认为平稳（两个检验结论相反）\n"
                   f"  可能原因: (1)序列处于临界状态 (2)窗口大小不当 (3)统计功效不足\n"
                   f"  建议: 结合Hurst({hurst:.3f})倾向于有趋势性，但需谨慎对待，建议降低仓位或尝试不同窗口")

        elif hurst < 0.5 and adf_nonstationary and kpss_nonstationary:
            # 矛盾2: 两个检验都说非平稳，但Hurst显示反趋势
            return (f"【矛盾-可能处于趋势转折】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  ⚠️ 矛盾分析: 检验显示非平稳但Hurst显示反持续性（可能正在转折）\n"
                   f"  可能原因: (1)趋势正在反转 (2)序列包含结构突变 (3)窗口包含多个趋势阶段\n"
                   f"  建议: 不建议交易，等待趋势明朗或使用更小窗口重新分析")

        elif not adf_nonstationary and kpss_nonstationary:
            # 矛盾3: ADF说平稳，KPSS说非平稳
            return (f"【矛盾-需进一步验证】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  ⚠️ 矛盾分析: ADF认为平稳但KPSS认为非平稳（两个检验结论相反，罕见情况）\n"
                   f"  可能原因: (1)序列包含去趋势后的平稳成分 (2)KPSS检测到线性趋势 (3)统计检验边界情况\n"
                   f"  建议: 高度不确定，不建议交易")

        else:
            # 其他情况
            adf_stat_str = "非平稳" if adf_nonstationary else "平稳"
            kpss_stat_str = "非平稳" if kpss_nonstationary else "平稳"

            return (f"【弱趋势或其他】\n"
                   f"  {hurst_str}\n"
                   f"  {hypothesis_explanation}\n"
                   f"  综合结论: ADF判断{adf_stat_str}, KPSS判断{kpss_stat_str}, "
                   f"Hurst={hurst:.3f}, 趋势特征不明显，不建议强趋势策略")

    def analyze_window(
        self,
        start_idx: int,
        end_idx: int,
        min_lag: Optional[int] = None,
        max_lag: Optional[int] = None,
    ) -> Dict:
        """
        对指定窗口进行三重检验分析

        Parameters:
        -----------
        start_idx: int
            起始索引
        end_idx: int
            结束索引（包含）
        min_lag: Optional[int]
            Hurst计算的最小滞后期
        max_lag: Optional[int]
            Hurst计算的最大滞后期

        Returns:
        --------
        dict: 包含三个检验结果和综合评分
        """
        window_size = end_idx - start_idx + 1

        # 自适应滞后参数
        if min_lag is None:
            min_lag = max(5, window_size // 6)
        if max_lag is None:
            max_lag = max(10, window_size // 3)

        # 确保max_lag合理
        max_lag = min(max_lag, window_size // 2)

        # 提取窗口数据
        window_prices = self.close_prices[start_idx : end_idx + 1]

        # 执行三个检验
        hurst = self.calculate_hurst(window_prices, min_lag, max_lag)
        adf_result = self.run_adf_test(window_prices)
        kpss_result = self.run_kpss_test(window_prices, regression="ct")

        # 计算综合评分
        trend_score = self._calculate_trend_score(hurst, adf_result, kpss_result)
        trend_type = self._classify_trend_type(hurst, adf_result, kpss_result)

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "window_size": window_size,
            "hurst": hurst,
            "adf": adf_result,
            "kpss": kpss_result,
            "trend_score": trend_score,
            "trend_type": trend_type,
            "min_lag": min_lag,
            "max_lag": max_lag,
        }

    def sliding_window_analysis(
        self,
        windows: List[int],
        step: int = 1,
        min_lag_func: Optional[Callable[[int], int]] = None,
        max_lag_func: Optional[Callable[[int], int]] = None,
    ) -> Dict[int, List[Dict]]:
        """
        滑动窗口批量分析

        Parameters:
        -----------
        windows: List[int]
            窗口大小列表
        step: int
            滑动步长
        min_lag_func: Optional[Callable]
            计算min_lag的函数，输入窗口大小
        max_lag_func: Optional[Callable]
            计算max_lag的函数，输入窗口大小

        Returns:
        --------
        Dict[int, List[Dict]] - 每个窗口大小的分析结果列表
        """
        if min_lag_func is None:
            min_lag_func = lambda w: max(5, w // 6)
        if max_lag_func is None:
            max_lag_func = lambda w: max(10, w // 3)

        total_data = len(self.close_prices)
        results = {}

        for window_size in windows:
            results[window_size] = []

            # 计算滞后参数
            min_lag = min_lag_func(window_size)
            max_lag = max_lag_func(window_size)
            window_max_lag = min(max_lag, window_size // 2)

            # 检查窗口大小是否有效
            if window_max_lag < min_lag:
                print(
                    f"窗口大小({window_size})过小，无法使用min_lag={min_lag}进行分析"
                )
                continue

            # 检查是否有足够数据
            if total_data < window_size:
                print(f"数据量({total_data})小于窗口大小({window_size})")
                continue

            # 滑动窗口循环
            for i in range(0, total_data - window_size + 1, step):
                start_idx = i
                end_idx = i + window_size - 1

                try:
                    window_result = self.analyze_window(
                        start_idx, end_idx, min_lag, window_max_lag
                    )
                    window_result["window_id"] = i // step + 1
                    results[window_size].append(window_result)

                except Exception as e:
                    print(f"分析窗口 {i//step + 1} 时出错: {e}")
                    continue

        self.results = results
        return results

    def get_summary_statistics(self) -> Dict[int, Dict]:
        """
        获取每个窗口大小的汇总统计
        """
        if not self.results:
            return {}

        summary = {}
        for window_size, window_results in self.results.items():
            if not window_results:
                continue

            scores = [r["trend_score"] for r in window_results]
            hursts = [r["hurst"] for r in window_results if not np.isnan(r["hurst"])]

            summary[window_size] = {
                "total_windows": len(window_results),
                "avg_trend_score": np.mean(scores),
                "median_trend_score": np.median(scores),
                "std_trend_score": np.std(scores),
                "avg_hurst": np.mean(hursts) if hursts else np.nan,
                "high_trend_ratio": sum(1 for s in scores if s >= 4) / len(scores),
                "low_trend_ratio": sum(1 for s in scores if s <= 2) / len(scores),
            }

        return summary
    
    def get_statistics_table(self) -> "pd.DataFrame":
        """
        生成各窗口趋势适配性评分统计表格
        
        Returns:
        --------
        pd.DataFrame: 包含各窗口统计信息的表格
        """
        import pandas as pd
        
        if not self.results:
            return pd.DataFrame()
        
        stats_list = []
        for window_size, window_results in self.results.items():
            if not window_results:
                continue
                
            scores = [r['trend_score'] for r in window_results]
            
            stats = {
                '窗口大小': window_size,
                '总窗口数': len(scores),
                '平均分': np.mean(scores),
                '中位数': np.median(scores),
                '最高分占比': (np.array(scores) == 5).mean() * 100,  # 5分占比
                '低分占比': (np.array(scores) <= 2).mean() * 100,    # ≤2分占比
                '评分标准差': np.std(scores)
            }
            stats_list.append(stats)
        
        # 创建DataFrame并按平均分降序排序
        stats_df = pd.DataFrame(stats_list)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('平均分', ascending=False)
            stats_df = stats_df.reset_index(drop=True)
        
        return stats_df
    
    def print_statistics_summary(self, save_to_file: Optional[str] = None) -> "pd.DataFrame":
        """
        打印Markdown格式的统计摘要，可选保存到.md文件
        
        Parameters:
        -----------
        save_to_file: Optional[str]
            如果提供，将结果保存到指定的.md文件路径
            
        Returns:
        --------
        pd.DataFrame: 统计表格
        """
        import pandas as pd
        stats_df = self.get_statistics_table()
        
        if stats_df.empty:
            print("没有可用的分析结果")
            return pd.DataFrame()
        
        # 构建Markdown格式输出
        output_lines = []
        output_lines.append("# 各窗口趋势适配性评分统计\n")
        
        # 创建Markdown表格
        output_lines.append("| 窗口大小 | 总窗口数 | 平均分 | 中位数 | 最高分占比 | 低分占比 | 评分标准差 |")
        output_lines.append("|:--------:|:--------:|:------:|:------:|:----------:|:--------:|:----------:|")
        
        # 添加数据行
        for _, row in stats_df.iterrows():
            line = (
                f"| {int(row['窗口大小'])} | "
                f"{int(row['总窗口数'])} | "
                f"{row['平均分']:.2f} | "
                f"{row['中位数']:.1f} | "
                f"{row['最高分占比']:.2f}% | "
                f"{row['低分占比']:.2f}% | "
                f"{row['评分标准差']:.2f} |"
            )
            output_lines.append(line)
        
        # 添加分析总结
        output_lines.append("\n## 分析总结\n")
        
        best_window = stats_df.iloc[0]
        worst_window = stats_df.iloc[-1]
        most_stable = stats_df.loc[stats_df['评分标准差'].idxmin()]
        highest_trend = stats_df.loc[stats_df['最高分占比'].idxmax()]
        
        output_lines.append(f"- **最佳窗口大小**: {int(best_window['窗口大小'])} （平均评分 {best_window['平均分']:.2f}）")
        output_lines.append(f"- **最差窗口大小**: {int(worst_window['窗口大小'])} （平均评分 {worst_window['平均分']:.2f}）")
        output_lines.append(f"- **最稳定窗口**: {int(most_stable['窗口大小'])} （标准差 {most_stable['评分标准差']:.2f}）")
        output_lines.append(f"- **高趋势占比最高**: {int(highest_trend['窗口大小'])} （5分占比 {highest_trend['最高分占比']:.1f}%）")
        
        # 打印结果
        output_text = '\n'.join(output_lines)
        print(output_text)
        
        # 保存到.md文件（如果指定）
        if save_to_file:
            # 确保文件名以.md结尾
            if not save_to_file.endswith('.md'):
                save_to_file = save_to_file.rsplit('.', 1)[0] + '.md'
            
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
                f.write("\n\n---\n\n")
                f.write("## 详细数据\n\n")
                # 手动创建Markdown表格，避免依赖tabulate
                f.write("| 窗口大小 | 总窗口数 | 平均分 | 中位数 | 最高分占比 | 低分占比 | 评分标准差 |\n")
                f.write("|:--------:|:--------:|:------:|:------:|:----------:|:--------:|:----------:|\n")
                for _, row in stats_df.iterrows():
                    f.write(
                        f"| {int(row['窗口大小'])} | "
                        f"{int(row['总窗口数'])} | "
                        f"{row['平均分']:.4f} | "
                        f"{row['中位数']:.2f} | "
                        f"{row['最高分占比']:.4f} | "
                        f"{row['低分占比']:.4f} | "
                        f"{row['评分标准差']:.4f} |\n"
                    )
            print(f"\n统计结果已保存到: {save_to_file}")
        
        return stats_df
    
    def get_results_dataframe(self) -> "pd.DataFrame":
        """
        将分析结果转换为Pandas DataFrame格式
        
        Returns:
        --------
        pd.DataFrame: 包含所有窗口分析结果的表格
        """
        import pandas as pd
        
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for window_size, window_results in self.results.items():
            for res in window_results:
                # 提取KPSS警告信息
                kpss_warning = res['kpss'].get('warning', '')
                
                # 构建行数据
                row = {
                    '窗口大小': window_size,
                    '窗口编号': res.get('window_id', 0),
                    '起始索引': res['start_idx'],
                    '结束索引': res['end_idx'],
                    'Hurst指数': res['hurst'],
                    'ADF统计量': res['adf']['statistic'],
                    'ADF p值': res['adf']['pvalue'],
                    'ADF滞后期': res['adf']['lags'],
                    'KPSS统计量': res['kpss']['statistic'],
                    'KPSS p值': res['kpss']['pvalue'],
                    'KPSS警告': kpss_warning,
                    '趋势适配性评分': res['trend_score'],
                    '趋势类型': res['trend_type'],
                    'min_lag': res['min_lag'],
                    '实际max_lag': res['max_lag']
                }
                
                # 添加时间戳信息（如果K线数据包含时间戳）
                if hasattr(self, 'candles') and self.candles.shape[1] > 0:
                    row['起始时间戳'] = self.candles[res['start_idx'], 0]
                    row['结束时间戳'] = self.candles[res['end_idx'], 0]
                
                rows.append(row)
        
        # 创建DataFrame并排序
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(['窗口大小', '窗口编号'])
            df = df.reset_index(drop=True)
        
        return df


def quick_trend_check(candles: np.ndarray, window: int = 60) -> Tuple[float, str, int]:
    """
    快速趋势检查函数

    Parameters:
    -----------
    candles: np.ndarray
        Jesse风格K线数据
    window: int
        分析窗口大小

    Returns:
    --------
    tuple: (hurst指数, 趋势类型, 趋势评分)
    """
    analyzer = TrendAnalyzer(candles)

    # 使用最近的数据
    if len(candles) >= window:
        result = analyzer.analyze_window(len(candles) - window, len(candles) - 1)
        return result["hurst"], result["trend_type"], result["trend_score"]
    else:
        result = analyzer.analyze_window(0, len(candles) - 1)
        return result["hurst"], result["trend_type"], result["trend_score"]
