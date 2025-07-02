"""
pysr风格的符号回归 - 优化版本。
这个版本直接将K线数据传递到Julia环境中，避免文件I/O操作。
"""

import numpy as np
import pandas as pd
from jesse.utils import numpy_candles_to_dataframe
from joblib import Parallel, delayed
from pysr import PySRRegressor

from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret_from_candles


def prepare_data():
    """准备训练数据"""
    # 加载数据
    candles = np.load("data/btc_1m.npy")
    candles = candles[candles[:, 5] > 0]
    df = numpy_candles_to_dataframe(candles)

    feature_and_label = []

    # label - 这里的y不重要，因为损失函数中不使用y
    label = pd.Series(
        np.zeros(len(df)), index=df.index, name="label"  # 使用零向量作为占位符
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
    y = df_features_and_label["label"].values[:-1]  # 仅作为占位符

    # 准备对应的K线数据
    candles_for_julia = candles[NA_MAX_NUM:-1]  # 匹配X的长度

    return X, y, cols, candles_for_julia


# 准备数据
X, y, cols, candles_for_julia = prepare_data()


# 创建一个包含所有必要Julia代码的字符串
# 这个版本会直接将K线数据作为常量嵌入到Julia代码中
def create_julia_loss_with_data(candles_data):
    """创建包含数据的Julia损失函数"""
    # 将numpy数组转换为Julia格式的字符串
    candles_str = (
        np.array2string(
            candles_data,
            separator=",",
            suppress_small=True,
            threshold=999999999,  # 使用大整数代替np.inf
            max_line_width=999999999,  # 使用大整数代替np.inf
        )
        .replace("\n", "")
        .replace(" ", "")
    )

    julia_code = f"""
# 嵌入的K线数据
const CANDLES_DATA = Float64{candles_str}
const CANDLES_SHAPE = ({candles_data.shape[0]}, {candles_data.shape[1]})

# Julia版本的build_bar_by_cumsum函数
function build_bar_by_cumsum(candles::Matrix{{Float64}}, condition::Vector{{Float64}}, threshold::Float64)
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
function compute_kurtosis(merged_bars::Matrix{{Float64}})
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
    
    # 计算峰度 (使用excess kurtosis)
    n = length(standardized)
    kurt = sum(standardized.^4) / n
    
    return kurt
end

# 主损失函数
function kurtosis_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
    # 获取预测值
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return L(1000)
    end
    
    # 检查预测值长度
    if length(y_pred) <= 2
        return L(1000)
    end
    
    # 从嵌入的数据中获取对应长度的K线数据
    candles = reshape(CANDLES_DATA, CANDLES_SHAPE)[1:length(y_pred), :]
    
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
    return julia_code


# 如果K线数据太大，使用更高效的方法
def create_julia_loss_simple():
    """创建简化版的Julia损失函数，使用外部文件"""
    return """
using DelimitedFiles
using Statistics

# 全局变量存储K线数据
global candles_data = nothing

# 加载K线数据的函数
function load_candles_data(filepath::String)
    global candles_data
    candles_data = readdlm(filepath, ',', Float64)
end

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
            bar_cumsum += condition[i]
            bar_timestamp = max(bar_timestamp, candles[i, 1])
            bar_volume += candles[i, 6]
            bar_high = max(bar_high, candles[i, 4])
            bar_low = min(bar_low, candles[i, 5])
            bar_close = candles[i, 3]
        else
            bars[bar_index, 1] = bar_timestamp
            bars[bar_index, 2] = bar_open
            bars[bar_index, 3] = bar_close
            bars[bar_index, 4] = bar_high
            bars[bar_index, 5] = bar_low
            bars[bar_index, 6] = bar_volume
            bar_index += 1
            
            bar_timestamp = candles[i, 1]
            bar_open = candles[i, 2]
            bar_close = candles[i, 3]
            bar_high = candles[i, 4]
            bar_low = candles[i, 5]
            bar_volume = candles[i, 6]
            bar_cumsum = condition[i]
        end
    end
    
    return bars[1:bar_index-1, :]
end

# 计算峰度
function compute_kurtosis(merged_bars::Matrix{Float64})
    n = size(merged_bars, 1)
    if n <= 5
        return 1000.0
    end
    
    close_prices = merged_bars[:, 3]
    log_returns = log.(close_prices[6:end] ./ close_prices[1:end-5])
    
    if length(log_returns) == 0
        return 1000.0
    end
    
    mean_ret = mean(log_returns)
    std_ret = std(log_returns)
    if std_ret == 0
        return 1000.0
    end
    
    standardized = (log_returns .- mean_ret) ./ std_ret
    n = length(standardized)
    kurt = sum(standardized.^4) / n
    
    return kurt
end

# 主损失函数
function kurtosis_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return L(1000)
    end
    
    if length(y_pred) <= 2
        return L(1000)
    end
    
    # 确保K线数据已加载
    if candles_data === nothing
        error("K线数据未加载，请先调用load_candles_data()")
    end
    
    candles = candles_data[1:length(y_pred), :]
    cumsum_threshold = sum(y_pred) / (length(y_pred) ÷ 120)
    merged_bars = build_bar_by_cumsum(candles, y_pred, cumsum_threshold)
    
    if size(merged_bars, 1) < length(y_pred) ÷ 240
        return L(1000)
    end
    
    kurtosis = compute_kurtosis(merged_bars)
    return L(kurtosis)
end
"""


# 保存K线数据为CSV格式
np.savetxt("candles_for_julia.csv", candles_for_julia, delimiter=",")

# 使用简化版Julia损失函数
julia_loss = create_julia_loss_simple()

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
    turbo=True,  # 启用turbo模式以加速
    # 输出配置
    verbosity=1,
    progress=True,
    # 自定义损失函数
    loss=julia_loss,
    # Julia启动命令 - 在启动时加载K线数据
    julia_kwargs=dict(
        startup_commands=[
            'include("load_candles.jl")',  # 需要创建这个文件
        ]
    ),
)

# 创建Julia启动脚本
with open("load_candles.jl", "w") as f:
    f.write(
        """
# 自动加载K线数据
println("正在加载K线数据...")
load_candles_data("candles_for_julia.csv")
println("K线数据加载完成，形状: ", size(candles_data))
"""
    )

# 使用说明
print("优化版本使用说明：")
print("1. 直接运行即可，K线数据会自动加载")
print("2. 运行训练：")
print("   model.fit(X, y)")
print("3. 查看结果：")
print("   print(model.get_best())")
print("   print(model.equations_)")
print("\n注意：")
print("- candles_for_julia.csv 文件已生成")
print("- load_candles.jl 文件已生成")
print("- 如果出现路径问题，请使用绝对路径")


# 额外的辅助函数 - 用于验证结果
def evaluate_expression(model, X, candles_data, expr_index=None):
    """评估某个表达式的峰度"""
    from scipy import stats

    from custom_indicators.toolbox.bar.build import build_bar_by_cumsum

    if expr_index is None:
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X, index=expr_index)

    cumsum_threshold = np.sum(y_pred) / (len(y_pred) // 120)
    merged_bars = build_bar_by_cumsum(
        candles_data,
        y_pred,
        cumsum_threshold,
        reverse=False,
    )

    if len(merged_bars) < len(candles_data) // 240:
        return 1000.0, len(merged_bars)

    close_arr = merged_bars[:, 2]
    if len(close_arr) <= 5:
        return 1000.0, len(merged_bars)

    ret = np.log(close_arr[5:] / close_arr[:-5])
    standard = (ret - ret.mean()) / ret.std()
    kurtosis = stats.kurtosis(standard, fisher=False, nan_policy="omit")

    return kurtosis, len(merged_bars)


# 保存评估函数供后续使用
print("\n评估函数使用示例：")
print("kurtosis, n_bars = evaluate_expression(model, X, candles_for_julia)")
print("print(f'峰度: {kurtosis:.4f}, 合并后K线数: {n_bars}')")
