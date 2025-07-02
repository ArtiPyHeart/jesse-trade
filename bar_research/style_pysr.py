"""
pysr风格的符号回归 - 完全Julia实现版本。
目标：找到分布最接近正态分布的表达式。
评估方式：通过评估整合后的bar的kurtosis间接评估。
本文件代码仅仅作为示例，实际运行在jupyter notebook中进行。

参考：
https://github.com/jesse-ai/jesse/blob/main/bar_research/style_gplearn.py
"""

import numpy as np
import pandas as pd
from jesse.utils import numpy_candles_to_dataframe
from joblib import Parallel, delayed
from pysr import PySRRegressor

from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret_from_candles

# Julia自定义损失函数 - 完全在Julia中实现
julia_kurtosis_loss = """
# Julia版本的build_bar_by_cumsum函数
function build_bar_by_cumsum(candles::Matrix{Float64}, condition::Vector{Float64}, threshold::Float64)
    n = size(candles, 1)
    bars = zeros(Float64, n, 6)
    bar_index = 1
    
    # 初始化第一个bar
    bar_timestamp = candles[1, 1]
    bar_open = candles[1, 2]
    bar_close = candles[1, 3]
    bar_high = candles[1, 4]
    bar_low = candles[1, 5]
    bar_volume = candles[1, 6]
    bar_cumsum = condition[1]
    
    for i in 2:n
        if bar_cumsum <= threshold
            # 继续累积当前bar
            bar_cumsum += condition[i]
            bar_timestamp = max(bar_timestamp, candles[i, 1])
            bar_volume += candles[i, 6]
            bar_high = max(bar_high, candles[i, 4])
            bar_low = min(bar_low, candles[i, 5])
            bar_close = candles[i, 3]  # 更新收盘价
        else
            # 保存当前bar
            bars[bar_index, 1] = bar_timestamp
            bars[bar_index, 2] = bar_open
            bars[bar_index, 3] = bar_close
            bars[bar_index, 4] = bar_high
            bars[bar_index, 5] = bar_low
            bars[bar_index, 6] = bar_volume
            bar_index += 1
            
            # 重置为新bar
            bar_timestamp = candles[i, 1]
            bar_open = candles[i, 2]
            bar_close = candles[i, 3]
            bar_high = candles[i, 4]
            bar_low = candles[i, 5]
            bar_volume = candles[i, 6]
            bar_cumsum = condition[i]
        end
    end
    
    # 返回有效的bars
    return bars[1:bar_index-1, :]
end

# 计算峰度的函数
function compute_kurtosis(merged_bars::Matrix{Float64})
    n = size(merged_bars, 1)
    if n <= 5
        return 1000.0
    end
    
    # 提取收盘价
    close_prices = merged_bars[:, 3]
    
    # 计算5期对数收益率
    log_returns = log.(close_prices[6:end] ./ close_prices[1:end-5])
    
    if length(log_returns) == 0
        return 1000.0
    end
    
    # 标准化
    mean_ret = mean(log_returns)
    std_ret = std(log_returns)
    if std_ret == 0
        return 1000.0
    end
    
    standardized = (log_returns .- mean_ret) ./ std_ret
    
    # 计算峰度 (使用excess kurtosis，即减去3)
    n = length(standardized)
    kurt = sum(standardized.^4) / n
    
    return kurt
end

# 主损失函数
function kurtosis_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # 获取预测值
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return L(1000)
    end
    
    # 检查预测值长度
    if length(y_pred) <= 2
        return L(1000)
    end
    
    # 从预定义的全局变量获取K线数据
    # 注意：candles_data需要在Julia环境中预先定义
    candles = candles_data[1:length(y_pred), :]
    
    # 计算累积阈值
    cumsum_threshold = sum(y_pred) / (length(y_pred) ÷ 120)
    
    # 构建合并后的K线
    merged_bars = build_bar_by_cumsum(candles, y_pred, cumsum_threshold)
    
    # 检查合并后的K线数量
    if size(merged_bars, 1) < length(y_pred) ÷ 240
        return L(1000)
    end
    
    # 计算峰度
    kurtosis = compute_kurtosis(merged_bars)
    
    return L(kurtosis)
end
"""


# 准备数据的Python代码
def prepare_data_and_julia_setup():
    """准备数据并生成Julia初始化代码"""
    # 加载数据
    candles = np.load("data/btc_1m.npy")
    candles = candles[candles[:, 5] > 0]
    df = numpy_candles_to_dataframe(candles)

    feature_and_label = []

    # label
    label = pd.Series(
        np.log(df["close"].shift(-1) / df["close"]), index=df.index, name="label"
    )
    feature_and_label.append(label)

    # high low range
    hl_range = pd.Series(
        np.log(df["high"] / df["low"]), index=df.index, name="hl_range"
    )
    feature_and_label.append(hl_range)

    RANGE = [25, 50, 100, 200]

    # log return
    for i in RANGE:
        series = pd.Series(
            np.log(df["close"] / df["close"].shift(i)), index=df.index, name=f"r{i}"
        )
        feature_and_label.append(series)

    # entropy
    for i in RANGE:
        log_ret_list = log_ret_from_candles(candles, [i] * len(candles))
        entropy_array = list(
            Parallel(n_jobs=-1)(delayed(sample_entropy_numba)(i) for i in log_ret_list)
        )
        len_gap = len(df) - len(entropy_array)
        entropy_array = [np.nan] * len_gap + entropy_array
        entropy_series = pd.Series(entropy_array, index=df.index, name=f"r{i}_entropy")
        feature_and_label.append(entropy_series)

    df_features_and_label = pd.concat(feature_and_label, axis=1)
    NA_MAX_NUM = df_features_and_label.isna().sum().max()
    df_features_and_label = df_features_and_label.iloc[NA_MAX_NUM:]

    cols = [col for col in df_features_and_label.columns if col != "label"]
    X = df_features_and_label[cols].values[:-1]
    y = df_features_and_label["label"].values[:-1]

    # 准备对应的K线数据
    candles_for_julia = candles[NA_MAX_NUM:-1]  # 匹配X的长度

    # 生成Julia初始化代码
    julia_setup_code = f"""
    # 在Julia环境中运行此代码以加载K线数据
    using NPZ
    using Statistics
    
    # 加载K线数据
    candles_data = npzread("candles_for_julia.npy")
    
    # 确保是Float64类型
    candles_data = Float64.(candles_data)
    
    # 验证数据形状
    println("Candles shape: ", size(candles_data))
    println("Expected shape: ({len(candles_for_julia)}, 6)")
    """

    # 保存K线数据供Julia使用
    np.save("candles_for_julia.npy", candles_for_julia)

    return X, y, cols, julia_setup_code


# 获取数据和Julia设置代码
X, y, cols, julia_setup_code = prepare_data_and_julia_setup()

# 创建PySR模型
model = PySRRegressor(
    # 模型配置
    niterations=30,
    populations=20,
    population_size=1000,
    # 操作符配置
    binary_operators=["plus", "sub", "max", "min"],
    unary_operators=["abs", "neg"],
    # 复杂度控制
    maxsize=20,
    parsimony=0.009,
    # 搜索参数
    tournament_selection_n=50,
    tournament_selection_p=0.86,
    # 变异概率
    crossover_probability=0.7,
    subtree_mutation_probability=0.12,
    hoist_mutation_probability=0.06,
    point_mutation_probability=0.12,
    # 优化配置
    optimize_probability=0.14,
    optimizer_nrestarts=2,
    optimizer_iterations=8,
    # 特征配置
    feature_names=cols,
    # 性能配置
    procs=12,
    multithreading=True,
    # 输出配置
    verbosity=1,
    progress=True,
    # 自定义损失函数
    loss=julia_kurtosis_loss,
    # Julia环境设置
    julia_project=None,  # 使用默认Julia环境
    extra_sympy_mappings={},
    extra_torch_mappings={},
    extra_jax_mappings={},
)

# 使用说明
print("使用步骤：")
print("1. 首先在Julia REPL中运行以下代码来设置环境：")
print(julia_setup_code)
print("\n2. 然后在Python中运行：")
print("model.fit(X, y)")
print("\n3. 查看结果：")
print("print(model.get_best())")
print("print(model.latex())")

# 注意事项
"""
重要提示：
1. 运行前需要确保Julia已安装以下包：
   - NPZ (用于读取numpy文件): `using Pkg; Pkg.add("NPZ")`
   - Statistics (标准库，通常已包含)

2. 由于PySR会在Julia中运行自定义损失函数，需要确保：
   - candles_for_julia.npy文件已生成并在正确路径
   - Julia环境可以访问该文件

3. 如果遇到问题，可以尝试：
   - 在Julia REPL中手动测试损失函数
   - 检查Julia的工作目录是否正确
   - 使用绝对路径加载npy文件

4. 优化建议：
   - 可以通过调整population_size和populations来平衡速度和质量
   - 如果损失函数运行缓慢，可以考虑减少K线合并的计算复杂度
"""
