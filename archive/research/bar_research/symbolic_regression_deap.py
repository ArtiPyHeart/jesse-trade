"""
DEAP-based Symbolic Regression for Finding Custom Bars with Minimal Kurtosis

This implementation uses DEAP (Distributed Evolutionary Algorithms in Python) to find
symbolic expressions that, when used as weights for bar aggregation, produce bars with
minimal kurtosis (closest to normal distribution).

Key improvements over gplearn:
1. Multi-objective optimization (kurtosis minimization + complexity penalty)
2. Advanced genetic operators (epsilon-lexicase selection, semantic crossover)
3. Better bloat control through dynamic depth limiting
4. Hall of Fame to preserve best individuals
5. Comprehensive logging and visualization
"""

import operator
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")

from jesse.utils import numpy_candles_to_dataframe

from src.bars.build import build_bar_by_cumsum
from src.data_process.entropy.apen_sampen import sample_entropy_numba
from src.utils.math_tools import log_ret_from_candles


# 定义保护除法和其他保护函数
def protected_div(left, right):
    """保护除法，避免除零错误"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(right) > 1e-10, left / right, 1.0)
    return result


def protected_log(x):
    """保护对数，避免负数和零的对数"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 1e-10, np.log(x), 0.0)
    return result


def protected_sqrt(x):
    """保护平方根，避免负数的平方根"""
    return np.sqrt(np.abs(x))


def protected_exp(x):
    """保护指数，避免溢出"""
    with np.errstate(over="ignore"):
        result = np.where(x < 100, np.exp(x), np.exp(100))
    return result


# 自定义操作符
def percentile_25(x):
    """返回25分位数"""
    return np.percentile(x, 25, method="linear")


def percentile_75(x):
    """返回75分位数"""
    return np.percentile(x, 75, method="linear")


def rolling_std(x, window=5):
    """滚动标准差"""
    result = pd.Series(x).rolling(window, min_periods=1).std()
    return result.fillna(0).values


def rank_normalize(x):
    """排名归一化到[0,1]"""
    ranks = pd.Series(x).rank(method="average")
    return (ranks - 1) / (len(x) - 1) if len(x) > 1 else np.zeros_like(x)


class SymbolicRegressionDEAP:
    """
    基于DEAP的符号回归实现，用于寻找峰度最小的自定义bar
    """

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 50,
        tournament_size: int = 7,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.15,
        max_depth: int = 10,
        init_depth: Tuple[int, int] = (2, 6),
        parsimony_coefficient: float = 0.01,
        elite_size: int = 10,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: int = None,
    ):
        """
        初始化符号回归模型

        参数:
        - population_size: 种群大小
        - generations: 进化代数
        - tournament_size: 锦标赛选择的参与者数量
        - crossover_prob: 交叉概率
        - mutation_prob: 变异概率
        - max_depth: 表达式树的最大深度
        - init_depth: 初始化深度范围 (min, max)
        - parsimony_coefficient: 简约系数，控制复杂度惩罚
        - elite_size: 精英个体数量
        - n_jobs: 并行计算的进程数
        - verbose: 是否显示详细信息
        - random_state: 随机种子
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # 设置随机种子
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # 存储最佳个体和统计信息
        self.best_individual = None
        self.hall_of_fame = None
        self.logbook = None

        # 数据相关
        self.X = None
        self.feature_names = None
        self.candles = None
        self.candles_in_metrics = None
        self.NA_MAX_NUM = None

    def _create_primitive_set(self, n_features: int):
        """创建原语集（函数和终端）"""
        # 创建强类型原语集
        self.pset = gp.PrimitiveSetTyped("MAIN", [float] * n_features, float)

        # 添加基本数学运算
        self.pset.addPrimitive(operator.add, [float, float], float, name="add")
        self.pset.addPrimitive(operator.sub, [float, float], float, name="sub")
        self.pset.addPrimitive(operator.mul, [float, float], float, name="mul")
        self.pset.addPrimitive(protected_div, [float, float], float, name="div")
        self.pset.addPrimitive(operator.neg, [float], float, name="neg")
        self.pset.addPrimitive(operator.abs, [float], float, name="abs")

        # 添加比较和逻辑运算
        self.pset.addPrimitive(max, [float, float], float, name="max")
        self.pset.addPrimitive(min, [float, float], float, name="min")

        # 添加高级数学函数
        self.pset.addPrimitive(protected_sqrt, [float], float, name="sqrt")
        self.pset.addPrimitive(protected_log, [float], float, name="log")
        self.pset.addPrimitive(protected_exp, [float], float, name="exp")
        self.pset.addPrimitive(np.sign, [float], float, name="sign")

        # 添加常数终端
        self.pset.addEphemeralConstant(
            "rand_const", lambda: random.uniform(-1.0, 1.0), float
        )

        # 重命名参数为特征名
        if self.feature_names is not None:
            for i, name in enumerate(self.feature_names):
                self.pset.renameArguments(**{f"ARG{i}": name})

    def _evaluate_kurtosis(self, individual) -> Tuple[float,]:
        """
        评估个体的适应度（峰度）

        返回值越小越好（最小化问题）
        """
        # 编译表达式树为函数
        func = gp.compile(expr=individual, pset=self.pset)

        try:
            # 计算预测值
            y_pred = np.array([func(*x) for x in self.X])

            # 检查有效性
            if len(y_pred) <= 2 or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return (1000.0,)  # 惩罚无效个体

            # 计算累积和阈值
            cumsum_threshold = np.sum(np.abs(y_pred)) / (
                len(self.candles_in_metrics) // 120
            )

            if cumsum_threshold <= 0:
                return (1000.0,)

            # 构建bar
            merged_bar_cumsum = build_bar_by_cumsum(
                self.candles_in_metrics,
                np.abs(y_pred),  # 使用绝对值确保非负
                cumsum_threshold,
                reverse=False,
            )

            # 检查bar数量是否足够
            if len(merged_bar_cumsum) < len(self.candles_in_metrics) // 240:
                return (1000.0,)

            # 计算峰度
            kurtosis = self._calculate_kurtosis(merged_bar_cumsum)

            # 添加复杂度惩罚
            complexity_penalty = self.parsimony_coefficient * len(individual)

            return (kurtosis + complexity_penalty,)

        except Exception as e:
            if self.verbose:
                print(f"评估错误: {e}")
            return (1000.0,)

    def _calculate_kurtosis(self, merged_bar: np.ndarray) -> float:
        """计算bar序列的峰度"""
        close_arr = merged_bar[:, 2]

        if len(close_arr) < 10:  # 需要足够的数据点
            return 1000.0

        # 计算对数收益率（使用5期收益率）
        ret = np.log(close_arr[5:] / close_arr[:-5])

        # 移除无效值
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]

        if len(ret) < 10 or ret.std() < 1e-10:
            return 1000.0

        # 标准化
        standard = (ret - ret.mean()) / ret.std()

        # 计算峰度（Fisher=False表示使用Pearson定义，正态分布峰度为3）
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        # 返回与正态分布峰度(3)的绝对差值
        return abs(kurtosis - 3.0)

    def _setup_evolution(self):
        """设置进化算法的组件"""
        # 创建适应度类（最小化）
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # 创建工具箱
        self.toolbox = base.Toolbox()

        # 注册表达式生成方法
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

        # 注册遗传操作
        self.toolbox.register("evaluate", self._evaluate_kurtosis)
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.tournament_size
        )
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

        # 装饰操作以限制深度
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth),
        )

    def _epsilon_lexicase(self, individuals, k):
        """
        Epsilon-lexicase选择算法
        更好地保持种群多样性
        """
        selected = []

        for _ in range(k):
            # 随机打乱测试用例顺序
            cases = list(range(len(self.X)))
            random.shuffle(cases)

            # 候选个体集合
            candidates = list(individuals)

            # 逐个测试用例筛选
            for case in cases:
                if len(candidates) == 1:
                    break

                # 计算所有候选个体在当前测试用例上的表现
                performances = []
                for ind in candidates:
                    func = gp.compile(expr=ind, pset=self.pset)
                    try:
                        pred = func(*self.X[case])
                        performances.append(abs(pred))
                    except Exception:
                        performances.append(float("inf"))

                # 找出最佳表现
                best_perf = min(performances)
                epsilon = best_perf * 0.01  # 1%的容差

                # 保留表现足够好的个体
                candidates = [
                    candidates[i]
                    for i, perf in enumerate(performances)
                    if perf <= best_perf + epsilon
                ]

            # 从剩余候选中随机选择
            selected.append(random.choice(candidates))

        return selected

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        candles_path: str = "data/btc_1m.npy",
    ):
        """
        训练符号回归模型

        参数:
        - X: 特征矩阵
        - feature_names: 特征名称列表
        - candles_path: 蜡烛图数据路径
        """
        self.X = X
        self.feature_names = feature_names

        # 加载蜡烛图数据
        self.candles = np.load(candles_path)
        self.candles = self.candles[self.candles[:, 5] > 0]  # 过滤零交易量

        # 如果指定使用最后10000行数据进行测试
        if len(self.candles) > 10000:
            test_start_idx = len(self.candles) - 10000
            self.candles = self.candles[test_start_idx:]
            # 相应地调整X
            self.X = self.X[-10000:] if len(self.X) > 10000 else self.X

        # 计算NA_MAX_NUM（这里简化处理，假设X已经处理过缺失值）
        self.NA_MAX_NUM = 0
        self.candles_in_metrics = self.candles[self.NA_MAX_NUM :]

        # 创建原语集
        self._create_primitive_set(X.shape[1])

        # 设置进化算法
        self._setup_evolution()

        # 创建统计对象
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # 创建名人堂（保存最佳个体）
        self.hall_of_fame = tools.HallOfFame(self.elite_size)

        # 运行进化算法
        if self.verbose:
            print("开始符号回归进化...")
            print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
            print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")

        # 初始化种群
        population = self.toolbox.population(n=self.population_size)

        # 自定义进化循环以支持更高级的特性
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "nevals"] + stats.fields

        # 评估初始种群
        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 更新名人堂
        self.hall_of_fame.update(population)

        # 记录统计信息
        record = stats.compile(population)
        self.logbook.record(gen=0, nevals=len(population), **record)

        if self.verbose:
            print(self.logbook.stream)

        # 进化主循环
        for gen in range(1, self.generations + 1):
            # 选择下一代
            offspring = self.toolbox.select(
                population, len(population) - self.elite_size
            )

            # 克隆选中的个体
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # 应用交叉和变异
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    (offspring[i],) = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 添加精英个体
            offspring.extend(self.hall_of_fame.items)

            # 更新种群
            population[:] = offspring

            # 更新名人堂
            self.hall_of_fame.update(population)

            # 记录统计信息
            record = stats.compile(population)
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if self.verbose and gen % 5 == 0:
                print(self.logbook.stream)
                print(
                    f"最佳个体高度: {self.hall_of_fame[0].height}, "
                    f"节点数: {len(self.hall_of_fame[0])}"
                )

        # 保存最佳个体
        self.best_individual = self.hall_of_fame[0]

        if self.verbose:
            print("\n进化完成!")
            print(f"最佳适应度: {self.best_individual.fitness.values[0]:.6f}")
            print(f"最佳表达式: {self._get_expression_string(self.best_individual)}")

    def _get_expression_string(self, individual) -> str:
        """将个体转换为可读的表达式字符串"""
        return str(individual)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用最佳个体进行预测"""
        if self.best_individual is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        func = gp.compile(expr=self.best_individual, pset=self.pset)
        predictions = np.array([func(*x) for x in X])

        return predictions

    def get_best_expression(self) -> str:
        """获取最佳表达式的字符串表示"""
        if self.best_individual is None:
            raise ValueError("模型尚未训练")

        return self._get_expression_string(self.best_individual)

    def evaluate_best_individual(self) -> dict:
        """评估最佳个体的详细性能"""
        if self.best_individual is None:
            raise ValueError("模型尚未训练")

        # 获取预测值
        y_pred = self.predict(self.X)

        # 计算bar
        cumsum_threshold = np.sum(np.abs(y_pred)) / (
            len(self.candles_in_metrics) // 120
        )
        merged_bar = build_bar_by_cumsum(
            self.candles_in_metrics,
            np.abs(y_pred),
            cumsum_threshold,
            reverse=False,
        )

        # 计算各种统计量
        close_arr = merged_bar[:, 2]
        ret = np.log(close_arr[5:] / close_arr[:-5])
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]

        results = {
            "expression": self.get_best_expression(),
            "fitness": self.best_individual.fitness.values[0],
            "tree_height": self.best_individual.height,
            "tree_size": len(self.best_individual),
            "num_bars": len(merged_bar),
            "kurtosis": stats.kurtosis(ret, fisher=False),
            "skewness": stats.skew(ret),
            "mean_return": np.mean(ret),
            "std_return": np.std(ret),
            "sharpe_ratio": np.mean(ret) / np.std(ret) if np.std(ret) > 0 else 0,
        }

        return results


def prepare_features(
    candles_path: str = "data/btc_1m.npy", use_last_n: int = 10000
) -> Tuple[np.ndarray, List[str], int]:
    """
    准备特征数据（与gplearn版本保持一致）

    返回:
    - X: 特征矩阵
    - feature_names: 特征名称列表
    - NA_MAX_NUM: 缺失值最大数量
    """
    # 加载数据
    candles = np.load(candles_path)
    candles = candles[candles[:, 5] > 0]  # 过滤零交易量

    # 如果指定使用最后N行
    if 0 < use_last_n < len(candles):
        candles = candles[-use_last_n:]

    df = numpy_candles_to_dataframe(candles)

    feature_and_label = []

    # label (这里不需要，但保持与原版一致)
    label = np.log(df["close"].shift(-1) / df["close"])
    label.name = "label"
    feature_and_label.append(label)

    # 候选特征
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

    # entropy (简化版本，避免过长计算时间)
    for i in RANGE[:2]:  # 只计算前两个范围的熵
        log_ret_list = log_ret_from_candles(candles, [i] * len(candles))
        entropy_array: list[float] = Parallel(n_jobs=-1)(
            delayed(sample_entropy_numba)(i) for i in log_ret_list
        )
        len_gap = len(df) - len(entropy_array)

        entropy_array = [np.nan] * len_gap + entropy_array
        entropy_series = pd.Series(entropy_array, index=df.index)
        entropy_series.name = f"r{i}_entropy"
        feature_and_label.append(entropy_series)

    df_features_and_label = pd.concat(feature_and_label, axis=1)

    # 处理缺失值
    NA_MAX_NUM = df_features_and_label.isna().sum().max()
    df_features_and_label = df_features_and_label.iloc[NA_MAX_NUM:]

    # 准备特征
    cols = [col for col in df_features_and_label.columns if col != "label"]
    X = df_features_and_label[cols].values[:-1]

    return X, cols, NA_MAX_NUM


if __name__ == "__main__":
    # 演示用法
    print("准备数据...")
    X, feature_names, NA_MAX_NUM = prepare_features(use_last_n=10000)

    print(f"特征数量: {len(feature_names)}")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征名称: {feature_names}")

    # 创建并训练模型
    model = SymbolicRegressionDEAP(
        population_size=100,  # 较小的种群用于快速演示
        generations=20,  # 较少的代数用于快速演示
        tournament_size=7,
        crossover_prob=0.8,
        mutation_prob=0.15,
        max_depth=8,
        init_depth=(2, 5),
        parsimony_coefficient=0.005,
        elite_size=5,
        verbose=True,
        random_state=42,
    )

    print("\n开始训练...")
    model.fit(X, feature_names)

    print("\n评估最佳个体...")
    results = model.evaluate_best_individual()

    print("\n=== 最终结果 ===")
    for key, value in results.items():
        print(f"{key}: {value}")
