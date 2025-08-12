import numpy as np


def sigmoid_mapping(
    p_meta: float | np.ndarray, alpha=10.0, beta=0.5, min_size=0.0, max_size=1.0
):
    """
    将 meta label 概率 p_meta 映射到 [min_size, max_size] 区间的连续持仓比例，
    使用 Sigmoid 函数进行平滑过渡。

    参数:
    - p_meta   : float 或 numpy.ndarray, meta label 的预测概率 (0~1)
    - alpha    : float, 控制 Sigmoid 的陡峭度，数值越大曲线越陡
    - beta     : float, Sigmoid 的中心点，默认 0.5 即在 p_meta = 0.5 时输出曲线中点
    - min_size : float, 最小持仓比例
    - max_size : float, 最大持仓比例

    返回:
    - result : 与 p_meta 形状相同的映射结果, 取值在 [min_size, max_size] 之间
    """
    # 将输入转换为 numpy 数组, 便于同时处理标量或向量
    p_meta = np.array(p_meta, ndmin=1)

    # Sigmoid 函数
    sigmoid_val = 1.0 / (1.0 + np.exp(-alpha * (p_meta - beta)))

    # 映射到 [min_size, max_size]
    result = min_size + (max_size - min_size) * sigmoid_val

    # 如果只输入了标量, 则返回标量
    if result.size == 1:
        return float(result)
    return result


def power_mapping(p_meta: float | np.ndarray, gamma=2.0, threshold=0.5):
    """
    将 meta label 概率 p_meta 映射到 [0, 1] 区间的连续持仓比例，
    在 threshold 以下时持仓=0，threshold 以上采用幂函数进行放大。

    参数:
    - p_meta   : float 或 numpy.ndarray, meta label 的预测概率 (0~1)
    - gamma    : float, 幂指数, 控制曲线陡峭度, 数值越大越陡
    - threshold: float, 在此概率以下不持仓 (=0)

    返回:
    - result : 与 p_meta 形状相同的映射结果, 取值在 [0, 1] 之间
    """
    # 转换为 numpy 数组
    p_meta = np.array(p_meta, ndmin=1)

    # 幂函数映射
    # 若 p_meta <= threshold 则持仓=0；否则对 (p_meta - threshold)/(1 - threshold) 取 gamma 幂
    result = np.where(
        p_meta <= threshold,
        0.0,
        np.power((p_meta - threshold) / (1.0 - threshold), gamma),
    )

    # 如果只输入了标量, 则返回标量
    if result.size == 1:
        return float(result)
    return result


def discretize_position(new_pos: float | np.ndarray, old_pos=None, threshold=0.05):
    """
    将连续持仓比例离散化，并可选地根据阈值控制是否更新持仓。

    参数:
    - new_pos : float 或 numpy.ndarray
        映射后的“连续持仓比例”，如 0.32, 0.45 等。
    - step    : float, 持仓比例的最小离散步长, 例如0.1即将持仓限制在0,0.1,0.2,...,1.0。
    - old_pos : float 或 numpy.ndarray, 上一次实际持仓比例(可选)。
        - 如果提供，则会根据 threshold 判断是否进行调仓。
        - 如果不提供，则不做阈值比较，直接输出离散化后的持仓。
    - threshold : float, 调仓触发阈值(绝对差值).
        - 当 |discrete_new_pos - old_pos| < threshold 时，不执行调仓，保持原持仓；
        - 当 |discrete_new_pos - old_pos| >= threshold 时，执行调仓到新的离散化水平。

    返回:
    - final_pos : float 或 numpy.ndarray, 最终离散化后的持仓比例。
    """

    # 将输入统一为 numpy 数组，以便同时处理标量或向量
    new_pos_array = np.array(new_pos, ndmin=1)

    # # 1) 首先对新目标持仓做离散化：round 到 step 的整数倍
    # discrete_new_pos = np.round(new_pos_array / step) * step

    # 如果没有 old_pos，则直接返回离散化结果
    if old_pos is None:
        return new_pos_array.item() if new_pos_array.size == 1 else new_pos_array

    # 否则，需要和 old_pos 比较看是否超过阈值
    old_pos_array = np.array(old_pos, ndmin=1)
    # 如果 old_pos_array 与 discrete_new_pos 形状不同，这里要注意对齐处理，本示例略过

    # 2) 计算调仓差值
    diff = np.abs(new_pos_array - old_pos_array)
    # 3) 判断是否超出阈值
    mask_need_trade = diff >= threshold

    # 4) 如果差值小于 threshold，则保持原有仓位不变
    final_pos_array = np.where(mask_need_trade, new_pos_array, old_pos_array)

    # 根据输入类型返回标量或数组
    if final_pos_array.size == 1:
        return float(final_pos_array)
    return final_pos_array
