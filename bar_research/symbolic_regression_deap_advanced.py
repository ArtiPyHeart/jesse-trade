"""
Advanced DEAP-based Symbolic Regression with Multi-Objective Optimization

This advanced implementation includes:
1. NSGA-II multi-objective optimization (kurtosis vs complexity)
2. Semantic similarity-based crossover
3. Local search operators
4. Adaptive mutation rates
5. Island model for parallel evolution
6. Advanced bloat control mechanisms
7. Pareto front visualization
"""

import operator
import pickle
import random
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from jesse.utils import numpy_candles_to_dataframe

from custom_indicators.toolbox.bar.build import build_bar_by_cumsum


# 高级数学操作符
def protected_div(left, right):
    """保护除法"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(right) > 1e-10, left / right, 1.0)
    return result


def protected_log(x):
    """保护对数"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 1e-10, np.log(x), 0.0)
    return result


def protected_sqrt(x):
    """保护平方根"""
    return np.sqrt(np.abs(x))


def protected_exp(x):
    """保护指数"""
    with np.errstate(over="ignore"):
        result = np.where(x < 100, np.exp(x), np.exp(100))
    return result


def protected_pow(x, y):
    """保护幂运算"""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.where(
            np.abs(x) < 100, np.sign(x) * np.abs(x) ** np.abs(y), np.sign(x) * 100.0
        )
    return result


def moving_average(x, window=5):
    """移动平均"""
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    result = pd.Series(x).rolling(window, min_periods=1).mean()
    return result.values[0] if len(result) == 1 else result.values


def tanh_scaled(x):
    """缩放的双曲正切函数"""
    return np.tanh(x / 2.0)


def rand_const():
    """生成随机常数"""
    return random.uniform(-2.0, 2.0)


def if_then_else(condition, output1, output2):
    """条件运算符"""
    return output1 if condition > 0 else output2


# 全局变量用于多进程
_global_model_instance = None


def _init_worker(model_dict):
    """初始化工作进程"""
    global _global_model_instance

    # 在每个工作进程中重新创建DEAP类型
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # 重建模型实例
    _global_model_instance = AdvancedSymbolicRegressionDEAP.__new__(
        AdvancedSymbolicRegressionDEAP
    )
    for key, value in model_dict.items():
        setattr(_global_model_instance, key, value)


def _island_evolution_worker(args):
    """岛屿进化的工作函数"""
    island_id, population_data, generations = args

    model = _global_model_instance

    # 重建toolbox
    toolbox = base.Toolbox()

    # 注册操作
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=model.pset,
        min_=model.init_depth[0],
        max_=model.init_depth[1],
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评估函数
    toolbox.register("evaluate", model._evaluate_multi_objective)
    toolbox.register("select", tools.selNSGA2)

    # 注册遗传操作
    toolbox.register("mate", model._semantic_crossover)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=model.pset)
    toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")
    toolbox.register("local_search", model._local_search)

    # 深度限制
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=model.max_depth),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=model.max_depth),
    )

    # 重建种群
    population = []
    for ind_data in population_data:
        ind = creator.Individual(ind_data["tree"])
        if ind_data["fitness"] is not None:
            ind.fitness.values = ind_data["fitness"]
        population.append(ind)

    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # 执行岛屿进化
    final_pop, logbook = model._island_evolution_impl(
        island_id, population, toolbox, stats, generations
    )

    # 将种群转换为可序列化的格式
    serialized_pop = []
    for ind in final_pop:
        serialized_pop.append({"tree": list(ind), "fitness": ind.fitness.values})

    return serialized_pop, logbook


class AdvancedSymbolicRegressionDEAP:
    """
    高级DEAP符号回归实现，支持多目标优化和高级特性

    该类实现了基于DEAP的高级符号回归算法，主要特性包括：
    1. NSGA-II多目标优化：同时优化峰度偏差和表达式复杂度
    2. 岛屿模型并行进化：将种群分割到多个岛屿独立进化，定期迁移
    3. 语义相似度交叉：根据个体输出的相似度选择合适的交叉策略
    4. 局部搜索优化：对常数进行微调以改善性能
    5. 自适应变异率：根据进化进度和停滞情况动态调整变异率
    6. 高级膨胀控制：通过深度限制和复杂度惩罚控制表达式增长
    7. Pareto前沿可视化：展示多目标优化的权衡关系
    """

    # 窗口大小常量
    CUSUM_WINDOW = 120  # CUSUM累积和窗口大小，用于计算合并阈值
    MAX_CUSUM_WINDOW = 360  # 最大CUSUM窗口，用于验证合并结果的合理性

    def __init__(
        self,
        population_size: int = 300,
        generations: int = 100,
        tournament_size: int = 7,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        max_depth: int = 12,
        init_depth: Tuple[int, int] = (2, 6),
        parsimony_coefficient: float = 0.001,
        elite_size: int = 20,
        n_islands: int = 4,
        migration_rate: float = 0.1,
        local_search_prob: float = 0.05,
        adaptive_mutation: bool = True,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: int = None,
    ):
        """
        初始化高级符号回归模型

        参数详解：
        -----------
        population_size : int, default=300
            种群大小，即每一代中个体的数量。
            - 推荐值：200-500
            - 较大的种群可以增加多样性，但会增加计算成本
            - 建议根据问题复杂度和计算资源调整

        generations : int, default=100
            进化的代数。
            - 推荐值：50-200
            - 更多代数可以获得更好的结果，但耗时更长
            - 可以通过early stopping机制提前终止

        tournament_size : int, default=7
            锦标赛选择的参与者数量。
            - 推荐值：5-10
            - 较大的值增加选择压力，可能导致早熟收敛
            - 较小的值保持多样性，但收敛速度慢

        crossover_prob : float, default=0.9
            交叉概率，控制两个个体交换基因的概率。
            - 推荐值：0.8-0.95
            - 高交叉率有助于探索解空间
            - 过高可能破坏优良基因组合

        mutation_prob : float, default=0.1
            变异概率，控制个体基因突变的概率。
            - 推荐值：0.05-0.2
            - 用于维持种群多样性，避免局部最优
            - 配合自适应变异率使用效果更好

        max_depth : int, default=12
            表达式树的最大深度。
            - 推荐值：10-17
            - 限制表达式复杂度，防止过拟合
            - 太小可能限制表达能力，太大增加计算成本

        init_depth : Tuple[int, int], default=(2, 6)
            初始化时表达式树的深度范围(最小深度, 最大深度)。
            - 推荐值：(2, 6) 或 (3, 8)
            - 影响初始种群的多样性
            - 较大范围增加初始多样性

        parsimony_coefficient : float, default=0.001
            简约系数，用于惩罚复杂表达式（已被多目标优化替代）。
            - 推荐值：0.0001-0.01
            - 在单目标优化中控制膨胀
            - 在多目标优化中可设为0

        elite_size : int, default=20
            精英个体数量，保留最佳个体到下一代。
            - 推荐值：10-50
            - 确保优良基因不丢失
            - 过大可能降低多样性

        n_islands : int, default=4
            岛屿模型中的岛屿数量。
            - 推荐值：2-8
            - 每个岛屿独立进化，增加搜索的并行性
            - 岛屿数量应与CPU核心数匹配以获得最佳性能

        migration_rate : float, default=0.1
            岛屿间个体迁移的比例。
            - 推荐值：0.05-0.2
            - 控制岛屿间基因交流的频率
            - 过高会降低岛屿独立性，过低限制基因流动

        local_search_prob : float, default=0.05
            对个体进行局部搜索优化的概率。
            - 推荐值：0.02-0.1
            - 微调常数以改善适应度
            - 过高会增加计算成本，过低效果不明显

        adaptive_mutation : bool, default=True
            是否使用自适应变异率。
            - 推荐值：True
            - 根据进化进度和停滞情况动态调整变异率
            - 早期高变异探索，后期低变异精调

        n_jobs : int, default=-1
            并行计算使用的CPU核心数。
            - -1：使用所有可用核心
            - 1：单线程执行（调试用）
            - n>1：使用n个核心

        verbose : bool, default=True
            是否输出详细的进化过程信息。
            - True：显示进化进度和统计信息
            - False：静默执行

        random_state : int, default=None
            随机数种子，用于结果复现。
            - None：使用系统时间作为种子
            - 整数：固定种子，确保结果可重复
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.init_depth = init_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.elite_size = elite_size
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.local_search_prob = local_search_prob
        self.adaptive_mutation = adaptive_mutation
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # 存储结果
        self.pareto_front = None
        self.best_individuals = None
        self.logbook = None
        self.semantic_cache = {}  # 缓存语义值以加速计算

        # 数据相关
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.feature_names = None
        self.candles = None
        self.candles_in_metrics = None
        self.NA_MAX_NUM = None

    def _create_advanced_primitive_set(self, n_features: int):
        """
        创建增强的原语集（Primitive Set）

        原语集定义了符号回归中可用的操作符、函数和终端节点。
        这是构建表达式树的基础组件。

        参数:
        ------
        n_features : int
            输入特征的数量，决定了终端节点（变量）的个数

        说明:
        ------
        - 基本运算：加减、取反、绝对值
        - 比较运算：最大值、最小值
        - 常数：随机常数、0、1
        - 部分高级函数被注释掉以降低搜索空间复杂度
        """
        self.pset = gp.PrimitiveSetTyped("MAIN", [float] * n_features, float)

        # 基本运算
        self.pset.addPrimitive(operator.add, [float, float], float, name="add")
        self.pset.addPrimitive(operator.sub, [float, float], float, name="sub")
        # self.pset.addPrimitive(operator.mul, [float, float], float, name="mul")
        # self.pset.addPrimitive(protected_div, [float, float], float, name="div")
        self.pset.addPrimitive(operator.neg, [float], float, name="neg")
        self.pset.addPrimitive(operator.abs, [float], float, name="abs")

        # 比较运算
        self.pset.addPrimitive(max, [float, float], float, name="max")
        self.pset.addPrimitive(min, [float, float], float, name="min")

        # 高级数学函数
        # self.pset.addPrimitive(protected_sqrt, [float], float, name="sqrt")
        # self.pset.addPrimitive(protected_log, [float], float, name="log")
        # self.pset.addPrimitive(protected_exp, [float], float, name="exp")
        # self.pset.addPrimitive(protected_pow, [float, float], float, name="pow")
        # self.pset.addPrimitive(np.sign, [float], float, name="sign")
        # self.pset.addPrimitive(tanh_scaled, [float], float, name="tanh")

        # 条件运算
        # self.pset.addPrimitive(if_then_else, [float, float, float], float, name="ite")

        # 常数
        self.pset.addEphemeralConstant("rand_const", rand_const, float)
        self.pset.addTerminal(0.0, float, name="zero")
        self.pset.addTerminal(1.0, float, name="one")

        # 重命名参数
        if self.feature_names is not None:
            for i, name in enumerate(self.feature_names):
                self.pset.renameArguments(**{f"ARG{i}": name})

    def _evaluate_multi_objective(self, individual) -> Tuple[float, float]:
        """
        多目标评估函数

        实现NSGA-II算法的多目标优化，同时考虑两个相互冲突的目标：
        1. 峰度偏差（kurtosis deviation）：衡量生成的bar序列的统计特性
        2. 表达式复杂度（complexity）：控制表达式的大小和深度

        参数:
        ------
        individual : gp.PrimitiveTree
            待评估的表达式树个体

        返回:
        ------
        Tuple[float, float]
            (峰度偏差, 复杂度) - 两个目标值，越小越好

        注意:
        ------
        - 使用语义缓存避免重复计算
        - 异常情况返回(1000.0, 1000.0)作为惩罚
        - 峰度计算基于5期对数收益率
        """
        # 获取缓存的语义值
        ind_str = str(individual)
        if ind_str in self.semantic_cache:
            y_pred = self.semantic_cache[ind_str]
        else:
            func = gp.compile(expr=individual, pset=self.pset)
            try:
                y_pred = np.array([func(*x) for x in self.X_scaled])
                self.semantic_cache[ind_str] = y_pred
            except Exception:
                return 1000.0, 1000.0

        # 计算峰度目标
        try:
            cumsum_threshold = np.sum(y_pred) / (
                len(self.candles_in_metrics) // self.CUSUM_WINDOW
            )

            # if cumsum_threshold <= 0:
            #     return 1000.0, 1000.0

            merged_bar = build_bar_by_cumsum(
                self.candles_in_metrics,
                np.abs(y_pred),
                cumsum_threshold,
                reverse=False,
            )

            if len(merged_bar) < len(self.candles_in_metrics) // self.MAX_CUSUM_WINDOW:
                return 1000.0, 1000.0

            kurtosis_deviation = self._calculate_kurtosis(merged_bar)

        except Exception:
            return 1000.0, 1000.0

        # 计算复杂度（考虑深度和大小）
        complexity = len(individual) + individual.height * 2

        return kurtosis_deviation, complexity

    def _calculate_kurtosis(self, merged_bar: np.ndarray, lag=3) -> float:
        """计算峰度偏差"""
        close_arr = merged_bar[:, 2]

        if len(close_arr) < 10:
            return 1000.0

        ret = np.log(close_arr[lag:] / close_arr[:-lag])
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]

        if len(ret) < 10 or ret.std() < 1e-10:
            return 1000.0

        standard = (ret - ret.mean()) / ret.std()
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        # 同时考虑峰度和偏度
        # skewness = abs(stats.skew(standard))

        # return abs(kurtosis - 3.0) + 0.1 * skewness
        return kurtosis

    def _semantic_crossover(self, ind1, ind2):
        """
        基于语义相似度的交叉操作

        这是一种智能交叉策略，根据两个个体的输出相似度来选择合适的交叉方式。
        相似的个体使用标准交叉，差异大的个体使用叶偏向交叉。

        参数:
        ------
        ind1, ind2 : gp.PrimitiveTree
            参与交叉的两个父代个体

        返回:
        ------
        Tuple[gp.PrimitiveTree, gp.PrimitiveTree]
            交叉后的两个子代个体

        策略:
        ------
        - 高相似度（语义接近）：使用标准单点交叉
        - 低相似度（语义差异大）：使用叶偏向交叉，倾向于交换叶节点
        - 使用余弦相似度衡量语义相似性
        """
        # 计算两个个体的语义输出
        func1 = gp.compile(expr=ind1, pset=self.pset)
        func2 = gp.compile(expr=ind2, pset=self.pset)

        try:
            # output1 = np.array([func1(*x) for x in self.X_scaled[:100]])  # 使用子集加速
            # output2 = np.array([func2(*x) for x in self.X_scaled[:100]])

            output1 = np.array([func1(*x) for x in self.X_scaled])
            output2 = np.array([func2(*x) for x in self.X_scaled])

            # 计算语义相似度
            if np.std(output1) > 0 and np.std(output2) > 0:
                similarity = 1 - cosine(output1, output2)
            else:
                similarity = 0

            # 根据相似度调整交叉概率
            if random.random() < similarity:
                # 相似度高时使用标准交叉
                return gp.cxOnePoint(ind1, ind2)
            else:
                # 相似度低时使用子树交叉
                return gp.cxOnePointLeafBiased(ind1, ind2, 0.1)

        except Exception:
            return gp.cxOnePoint(ind1, ind2)

    def _local_search(self, individual):
        """
        局部搜索操作，微调常数

        对表达式树中的数值常数进行局部优化，通过高斯扰动微调常数值。
        这是一种memetic算法的体现，结合全局搜索和局部优化。

        参数:
        ------
        individual : gp.PrimitiveTree
            待优化的个体

        返回:
        ------
        Tuple[gp.PrimitiveTree,]
            优化后的个体（返回元组以符合DEAP接口）

        策略:
        ------
        - 30%概率对每个常数节点进行扰动
        - 使用均值0、标准差0.1的高斯分布
        - 只修改浮点数常数，不影响变量和操作符
        """
        # 找到所有常数节点
        for i, node in enumerate(individual):
            if isinstance(node, float):
                # 以小概率微调常数
                if random.random() < 0.3:
                    # 高斯扰动
                    individual[i] = node + random.gauss(0, 0.1)

        return (individual,)

    def _adaptive_mutation_rate(
        self, generation: int, fitness_stagnation: int
    ) -> float:
        """
        自适应变异率

        根据进化进度和适应度停滞情况动态调整变异率。
        早期维持较高变异率以探索，后期降低变异率以精调。

        参数:
        ------
        generation : int
            当前进化代数
        fitness_stagnation : int
            适应度停滞的代数（多少代没有改进）

        返回:
        ------
        float
            调整后的变异率，范围[0, 0.5]

        策略:
        ------
        - 停滞5代：变异率×1.5
        - 停滞10代：变异率×2.0
        - 进化80%后：变异率×0.5（精细调整阶段）
        - 最大不超过0.5，避免过度破坏
        """
        base_rate = self.mutation_prob

        # 根据进化代数调整
        progress = generation / self.generations

        # 根据适应度停滞情况调整
        if fitness_stagnation > 5:
            rate_multiplier = 1.5
        elif fitness_stagnation > 10:
            rate_multiplier = 2.0
        else:
            rate_multiplier = 1.0

        # 后期降低变异率以精细调整
        if progress > 0.8:
            rate_multiplier *= 0.5

        return min(base_rate * rate_multiplier, 0.5)

    def _setup_multi_objective_evolution(self):
        """设置多目标进化算法"""
        # 创建多目标适应度类
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # 注册操作
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.init_depth[0],
            max_=self.init_depth[1],
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # 多目标评估
        self.toolbox.register("evaluate", self._evaluate_multi_objective)

        # NSGA-II选择
        self.toolbox.register("select", tools.selNSGA2)

        # 高级遗传操作
        self.toolbox.register("mate", self._semantic_crossover)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")
        self.toolbox.register("local_search", self._local_search)

        # 深度限制
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth),
        )

    def _island_evolution_impl(
        self,
        island_id: int,
        population: List,
        toolbox: base.Toolbox,
        stats: tools.Statistics,
        generations: int,
    ) -> Tuple[List, tools.Logbook]:
        """
        岛屿进化的实际实现

        每个岛屿独立进行遗传算法进化，包含选择、交叉、变异和局部搜索。
        这是岛屿模型的核心实现，支持并行执行。

        参数:
        ------
        island_id : int
            岛屿标识符
        population : List[Individual]
            岛屿的种群
        toolbox : base.Toolbox
            DEAP工具箱，包含遗传操作
        stats : tools.Statistics
            统计对象，记录进化过程
        generations : int
            进化代数

        返回:
        ------
        Tuple[List[Individual], tools.Logbook]
            (最终种群, 进化日志)

        算法流程:
        ---------
        1. 评估初始种群
        2. 对每一代：
           - 计算自适应变异率
           - NSGA-II选择
           - 交叉操作（语义交叉）
           - 变异操作（均匀变异或常数变异）
           - 局部搜索
           - 环境选择（保留最优个体）
        3. 检测适应度停滞并调整策略
        """
        logbook = tools.Logbook()
        logbook.header = ["gen", "island", "nevals"] + stats.fields

        # 评估初始种群
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 记录初始统计
        record = stats.compile(population)
        logbook.record(gen=0, island=island_id, nevals=len(population), **record)

        # 进化循环
        fitness_stagnation = 0
        best_fitness = float("inf")

        for gen in range(1, generations + 1):
            # 自适应变异率
            if self.adaptive_mutation:
                current_mutation_prob = self._adaptive_mutation_rate(
                    gen, fitness_stagnation
                )
            else:
                current_mutation_prob = self.mutation_prob

            # 选择
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 交叉和变异
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            for i in range(len(offspring)):
                if random.random() < current_mutation_prob:
                    if random.random() < 0.5:
                        (offspring[i],) = toolbox.mutate(offspring[i])
                    else:
                        (offspring[i],) = toolbox.mutate_eph(offspring[i])
                    del offspring[i].fitness.values

                # 局部搜索
                if random.random() < self.local_search_prob:
                    (offspring[i],) = toolbox.local_search(offspring[i])
                    del offspring[i].fitness.values

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 环境选择
            population[:] = toolbox.select(population + offspring, len(population))

            # 检查停滞
            current_best = min(population, key=lambda x: x.fitness.values[0])
            if current_best.fitness.values[0] < best_fitness:
                best_fitness = current_best.fitness.values[0]
                fitness_stagnation = 0
            else:
                fitness_stagnation += 1

            # 记录统计
            record = stats.compile(population)
            logbook.record(gen=gen, island=island_id, nevals=len(invalid_ind), **record)

        return population, logbook

    def _island_evolution(
        self,
        island_id: int,
        population: List,
        stats: tools.Statistics,
        generations: int,
    ) -> Tuple[List, tools.Logbook]:
        """
        岛屿进化（单进程版本，用于兼容性）
        """
        return self._island_evolution_impl(
            island_id, population, self.toolbox, stats, generations
        )

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        NA_MAX_NUM: int,
        stand_scale: bool = False,
        candles_path: str = "data/btc_1m.npy",
    ):
        """
        训练模型

        执行完整的符号回归训练流程，包括数据预处理、岛屿模型进化和结果收集。

        参数:
        ------
        X : np.ndarray
            输入特征矩阵，shape=(n_samples, n_features)
        feature_names : List[str]
            特征名称列表，用于生成可读的表达式
        NA_MAX_NUM : int
            最大缺失值数量，用于数据对齐
        stand_scale : bool, default=False
            是否对特征进行标准化处理
        candles_path : str, default="data/btc_1m.npy"
            K线数据文件路径

        训练流程:
        ---------
        1. 数据预处理：加载K线数据，特征标准化
        2. 创建原语集：定义可用的操作符和函数
        3. 设置多目标进化：配置NSGA-II算法
        4. 岛屿模型进化：
           - 创建多个岛屿种群
           - 并行或串行进化
           - 定期进行岛屿间迁移
        5. 收集Pareto前沿解

        注意:
        ------
        - 支持多进程并行加速
        - 每10代进行一次岛屿间迁移
        - 最终合并所有岛屿的种群
        """
        self.X = X
        self.feature_names = feature_names

        # 特征标准化
        if stand_scale:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
        else:
            self.X_scaled = X

        # 加载数据
        self.candles = np.load(candles_path)
        self.candles = self.candles[self.candles[:, 5] > 0]

        # if len(self.candles) > 10000:
        #     test_start_idx = len(self.candles) - 10000
        #     self.candles = self.candles[test_start_idx:]
        #     self.X = self.X[-10000:] if len(self.X) > 10000 else self.X
        #     self.X_scaled = (
        #         self.X_scaled[-10000:] if len(self.X_scaled) > 10000 else self.X_scaled
        #     )

        self.NA_MAX_NUM = NA_MAX_NUM
        self.candles_in_metrics = self.candles[self.NA_MAX_NUM :]

        # 创建原语集
        self._create_advanced_primitive_set(X.shape[1])

        # 设置进化
        self._setup_multi_objective_evolution()

        # 统计对象
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if self.verbose:
            print("开始多目标符号回归进化...")
            print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
            print(f"岛屿数量: {self.n_islands}, 迁移率: {self.migration_rate}")
            if self.n_jobs != 1 and self.n_islands > 1:
                n_processes = min(
                    self.n_islands, cpu_count() if self.n_jobs == -1 else self.n_jobs
                )
                print(f"使用多进程加速: {n_processes} 个进程")
            else:
                print("使用单进程模式")

        # 创建岛屿种群
        islands = []
        for i in range(self.n_islands):
            pop = self.toolbox.population(n=self.population_size // self.n_islands)
            islands.append(pop)

        # 主进化循环
        all_logbooks = []

        for gen in range(0, self.generations, 10):  # 每10代进行一次迁移
            # 决定是否使用多进程
            use_multiprocessing = self.n_jobs != 1 and self.n_islands > 1

            if use_multiprocessing:
                # 准备多进程所需的模型数据
                model_dict = {
                    "pset": self.pset,
                    "X": self.X,
                    "X_scaled": self.X_scaled,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "candles": self.candles,
                    "candles_in_metrics": self.candles_in_metrics,
                    "NA_MAX_NUM": self.NA_MAX_NUM,
                    "semantic_cache": self.semantic_cache,
                    "population_size": self.population_size,
                    "generations": self.generations,
                    "tournament_size": self.tournament_size,
                    "crossover_prob": self.crossover_prob,
                    "mutation_prob": self.mutation_prob,
                    "max_depth": self.max_depth,
                    "init_depth": self.init_depth,
                    "parsimony_coefficient": self.parsimony_coefficient,
                    "elite_size": self.elite_size,
                    "n_islands": self.n_islands,
                    "migration_rate": self.migration_rate,
                    "local_search_prob": self.local_search_prob,
                    "adaptive_mutation": self.adaptive_mutation,
                }

                # 序列化种群数据
                islands_data = []
                for island in islands:
                    island_data = []
                    for ind in island:
                        island_data.append(
                            {
                                "tree": list(ind),
                                "fitness": (
                                    ind.fitness.values if ind.fitness.valid else None
                                ),
                            }
                        )
                    islands_data.append(island_data)

                # 准备任务参数
                tasks = []
                for i, island_data in enumerate(islands_data):
                    tasks.append((i, island_data, min(10, self.generations - gen)))

                # 使用多进程池执行
                n_processes = min(
                    self.n_islands, cpu_count() if self.n_jobs == -1 else self.n_jobs
                )
                with Pool(
                    processes=n_processes,
                    initializer=_init_worker,
                    initargs=(model_dict,),
                ) as pool:
                    results = pool.map(_island_evolution_worker, tasks)

                # 更新岛屿种群和日志
                for i, (serialized_pop, logbook) in enumerate(results):
                    # 重建种群
                    new_pop = []
                    for ind_data in serialized_pop:
                        ind = creator.Individual(ind_data["tree"])
                        ind.fitness.values = ind_data["fitness"]
                        new_pop.append(ind)
                    islands[i] = new_pop
                    all_logbooks.append(logbook)
            else:
                # 单进程执行
                results = []
                for i, island in enumerate(islands):
                    result = self._island_evolution(
                        i, island, stats, min(10, self.generations - gen)
                    )
                    results.append(result)

                # 更新岛屿种群和日志
                for i, (pop, logbook) in enumerate(results):
                    islands[i] = pop
                    all_logbooks.append(logbook)

            # 岛屿间迁移
            if gen + 10 < self.generations:
                for i in range(self.n_islands):
                    # 选择迁移个体
                    migrants = tools.selBest(
                        islands[i], int(self.migration_rate * len(islands[i]))
                    )
                    # 迁移到下一个岛屿
                    next_island = (i + 1) % self.n_islands
                    # 替换最差个体
                    islands[next_island][-len(migrants) :] = migrants

            if self.verbose and gen % 20 == 0:
                print(f"Generation {gen}/{self.generations} completed")

        # 合并所有岛屿的种群
        final_population = []
        for island in islands:
            final_population.extend(island)

        # 获取Pareto前沿
        self.pareto_front = tools.sortNondominated(
            final_population, len(final_population), True
        )[0]

        # 保存最佳个体
        self.best_individuals = sorted(
            self.pareto_front, key=lambda x: x.fitness.values[0]
        )[: self.elite_size]

        if self.verbose:
            print(f"\n进化完成! Pareto前沿大小: {len(self.pareto_front)}")
            print(f"最佳峰度偏差: {self.best_individuals[0].fitness.values[0]:.6f}")
            print(f"对应复杂度: {self.best_individuals[0].fitness.values[1]:.0f}")

    def visualize_pareto_front(self, save_path: str = None):
        """
        可视化Pareto前沿

        绘制多目标优化的Pareto前沿图，展示峰度偏差与复杂度之间的权衡关系。
        Pareto前沿上的每个点代表一个非支配解，即不存在另一个解在所有目标上都优于它。

        参数:
        ------
        save_path : str, optional
            图片保存路径。如果提供，将保存图片到指定路径

        可视化内容:
        -----------
        - 所有Pareto前沿上的解（蓝色散点）
        - Top 5最优解（红色星号）
        - X轴：峰度偏差（越小越好）
        - Y轴：表达式复杂度（越小越好）

        抛出:
        ------
        ValueError
            如果模型尚未训练
        """
        if self.pareto_front is None:
            raise ValueError("模型尚未训练")

        plt.figure(figsize=(10, 6))

        # 提取目标值
        objectives = np.array([ind.fitness.values for ind in self.pareto_front])

        plt.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=50)
        plt.xlabel("Kurtosis Deviation")
        plt.ylabel("Complexity")
        plt.title("Pareto Front - Kurtosis vs Complexity")
        plt.grid(True, alpha=0.3)

        # 标记最佳几个解
        best_objectives = np.array(
            [ind.fitness.values for ind in self.best_individuals[:5]]
        )
        plt.scatter(
            best_objectives[:, 0],
            best_objectives[:, 1],
            color="red",
            s=100,
            marker="*",
            label="Best Solutions",
        )

        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def get_best_expressions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        获取最佳的n个表达式及其性能

        从Pareto前沿中选择最优的n个表达式，并计算它们的详细统计指标。
        选择标准是优先考虑峰度偏差最小的解。

        参数:
        ------
        n : int, default=5
            要返回的表达式数量

        返回:
        ------
        List[Dict[str, Any]]
            包含以下字段的字典列表：
            - rank: 排名
            - expression: 表达式字符串
            - kurtosis_deviation: 峰度偏差（优化目标）
            - complexity: 复杂度（优化目标）
            - height: 表达式树深度
            - size: 表达式节点数
            - num_bars: 生成的bar数量
            - actual_kurtosis: 实际峰度值
            - skewness: 偏度
            - sharpe_ratio: 夏普比率

        抛出:
        ------
        ValueError
            如果模型尚未训练
        """
        if self.best_individuals is None:
            raise ValueError("模型尚未训练")

        results = []
        for i, ind in enumerate(self.best_individuals[:n]):
            result = {
                "rank": i + 1,
                "expression": str(ind),
                "kurtosis_deviation": ind.fitness.values[0],
                "complexity": ind.fitness.values[1],
                "height": ind.height,
                "size": len(ind),
            }

            # 计算详细统计
            y_pred = self.predict(self.X, individual=ind)
            cumsum_threshold = np.sum(y_pred) / (
                len(self.candles_in_metrics) // self.CUSUM_WINDOW
            )
            merged_bar = build_bar_by_cumsum(
                self.candles_in_metrics,
                np.abs(y_pred),
                cumsum_threshold,
                reverse=False,
            )

            close_arr = merged_bar[:, 2]
            ret = np.log(close_arr[5:] / close_arr[:-5])
            ret = ret[~np.isnan(ret) & ~np.isinf(ret)]

            result.update(
                {
                    "num_bars": len(merged_bar),
                    "actual_kurtosis": stats.kurtosis(ret, fisher=False),
                    "skewness": stats.skew(ret),
                    "sharpe_ratio": (
                        np.mean(ret) / np.std(ret) if np.std(ret) > 0 else 0
                    ),
                }
            )

            results.append(result)

        return results

    def predict(self, X: np.ndarray, individual=None) -> np.ndarray:
        """
        预测

        使用训练得到的表达式对新数据进行预测。

        参数:
        ------
        X : np.ndarray
            输入特征矩阵，shape=(n_samples, n_features)
            特征顺序必须与训练时一致
        individual : gp.PrimitiveTree, optional
            指定使用的表达式个体。如果不提供，使用最佳个体

        返回:
        ------
        np.ndarray
            预测结果，shape=(n_samples,)

        抛出:
        ------
        ValueError
            如果模型尚未训练

        注意:
        ------
        - 如果训练时启用了特征标准化，输入数据会自动进行标准化
        - 预测结果是原始表达式的输出，通常用于计算bar合并阈值
        """
        if individual is None:
            if self.best_individuals is None:
                raise ValueError("模型尚未训练")
            individual = self.best_individuals[0]

        # 标准化输入
        X_scaled = self.scaler.transform(X) if self.scaler else X

        func = gp.compile(expr=individual, pset=self.pset)
        predictions = np.array([func(*x) for x in X_scaled])

        return predictions

    def save_model(self, filepath: str):
        """
        保存模型

        将训练得到的模型保存到文件，包括所有必要的组件以便后续加载和使用。

        参数:
        ------
        filepath : str
            保存文件的路径，建议使用.pkl扩展名

        保存内容:
        ---------
        - best_individuals: 最佳个体列表
        - pareto_front: 完整的Pareto前沿
        - pset: 原语集定义
        - scaler: 特征标准化器（如果使用）
        - feature_names: 特征名称
        - params: 模型参数
        """
        model_data = {
            "best_individuals": self.best_individuals,
            "pareto_front": self.pareto_front,
            "pset": self.pset,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "params": {
                "population_size": self.population_size,
                "generations": self.generations,
                "max_depth": self.max_depth,
                "n_islands": self.n_islands,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """
        加载模型

        从文件加载之前保存的模型，恢复所有必要的组件。

        参数:
        ------
        filepath : str
            模型文件的路径

        恢复内容:
        ---------
        - 最佳个体和Pareto前沿
        - 原语集和特征信息
        - 标准化器
        - 模型参数

        注意:
        ------
        加载后可以直接使用predict方法进行预测
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.best_individuals = model_data["best_individuals"]
        self.pareto_front = model_data["pareto_front"]
        self.pset = model_data["pset"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]

        for key, value in model_data["params"].items():
            setattr(self, key, value)


# 辅助函数保持与基础版本一致
def prepare_features(
    candles_path: str = "data/btc_1m.npy", use_last_n: int = 10000
) -> Tuple[np.ndarray, List[str], int]:
    """
    准备特征数据

    从K线数据中提取技术特征，用于符号回归训练。

    参数:
    ------
    candles_path : str, default="data/btc_1m.npy"
        K线数据文件路径
    use_last_n : int, default=10000
        使用最近n条K线数据。如果为0或负数，使用全部数据

    返回:
    ------
    Tuple[np.ndarray, List[str], int]
        - X: 特征矩阵，shape=(n_samples-1, n_features)
        - feature_names: 特征名称列表
        - NA_MAX_NUM: 最大缺失值数量，用于数据对齐

    特征包括:
    ----------
    - hl_range: 高低区间的对数，衡量波动性
    - r25, r50, r100, r200: 不同周期的对数收益率
    - vol_ratio: 成交量比率的对数
    - price_position: 价格在高低区间的位置

    注意:
    ------
    - 过滤掉成交量为0的K线
    - 移除有缺失值的前部行
    """
    candles = np.load(candles_path)
    candles = candles[candles[:, 5] > 0]

    if use_last_n > 0 and len(candles) > use_last_n:
        candles = candles[-use_last_n:]

    df = numpy_candles_to_dataframe(candles)

    feature_and_label = []

    # label
    label = np.log(df["close"].shift(-1) / df["close"])
    label.name = "label"
    feature_and_label.append(label)

    # high low range
    hl_range = np.log(df["high"] / df["low"])
    hl_range.name = "hl_range"
    feature_and_label.append(hl_range)

    RANGE = [25, 50, 100, 200]

    # log return
    for i in RANGE:
        series = np.log(df["close"] / df["close"].shift(i))
        series.name = f"r{i}"
        feature_and_label.append(series)

    # volume features
    vol_series = np.log(df["volume"] / df["volume"].shift(1) + 1)
    vol_series.name = "vol_ratio"
    feature_and_label.append(vol_series)

    # price position
    price_pos = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
    price_pos.name = "price_position"
    feature_and_label.append(price_pos)

    df_features_and_label = pd.concat(feature_and_label, axis=1)

    NA_MAX_NUM = df_features_and_label.isna().sum().max()
    df_features_and_label = df_features_and_label.iloc[NA_MAX_NUM:]

    cols = [col for col in df_features_and_label.columns if col != "label"]
    X = df_features_and_label[cols].values[:-1]

    return X, cols, NA_MAX_NUM


if __name__ == "__main__":
    # 高级版本演示
    print("准备数据...")
    X, feature_names, NA_MAX_NUM = prepare_features(use_last_n=10000)

    print(f"特征数量: {len(feature_names)}")
    print(f"样本数量: {X.shape[0]}")

    # 创建高级模型
    model = AdvancedSymbolicRegressionDEAP(
        population_size=200,
        generations=30,  # 演示用较少代数
        n_islands=4,
        migration_rate=0.1,
        local_search_prob=0.05,
        adaptive_mutation=True,
        verbose=True,
        random_state=42,
    )

    print("\n开始训练高级模型...")
    model.fit(X, feature_names)

    print("\n获取最佳表达式...")
    best_expressions = model.get_best_expressions(n=5)

    print("\n=== Top 5 表达式 ===")
    for expr in best_expressions:
        print(f"\n排名 {expr['rank']}:")
        print(f"  表达式: {expr['expression'][:100]}...")
        print(f"  峰度偏差: {expr['kurtosis_deviation']:.6f}")
        print(f"  复杂度: {expr['complexity']}")
        print(f"  实际峰度: {expr['actual_kurtosis']:.4f}")
        print(f"  Bar数量: {expr['num_bars']}")

    # 可视化Pareto前沿
    print("\n生成Pareto前沿可视化...")
    model.visualize_pareto_front(save_path="pareto_front.png")
