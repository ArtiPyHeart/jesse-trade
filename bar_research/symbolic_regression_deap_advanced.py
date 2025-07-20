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
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

# from joblib import Parallel, delayed  # Removed due to serialization issues
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


class AdvancedSymbolicRegressionDEAP:
    """
    高级DEAP符号回归实现，支持多目标优化和高级特性
    """

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

        新增参数:
        - n_islands: 岛屿模型中的岛屿数量
        - migration_rate: 岛屿间迁移率
        - local_search_prob: 局部搜索概率
        - adaptive_mutation: 是否使用自适应变异率
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
        """创建增强的原语集"""
        self.pset = gp.PrimitiveSetTyped("MAIN", [float] * n_features, float)

        # 基本运算
        self.pset.addPrimitive(operator.add, [float, float], float, name="add")
        self.pset.addPrimitive(operator.sub, [float, float], float, name="sub")
        self.pset.addPrimitive(operator.mul, [float, float], float, name="mul")
        self.pset.addPrimitive(protected_div, [float, float], float, name="div")
        self.pset.addPrimitive(operator.neg, [float], float, name="neg")
        self.pset.addPrimitive(operator.abs, [float], float, name="abs")

        # 比较运算
        self.pset.addPrimitive(max, [float, float], float, name="max")
        self.pset.addPrimitive(min, [float, float], float, name="min")

        # 高级数学函数
        self.pset.addPrimitive(protected_sqrt, [float], float, name="sqrt")
        self.pset.addPrimitive(protected_log, [float], float, name="log")
        self.pset.addPrimitive(protected_exp, [float], float, name="exp")
        self.pset.addPrimitive(protected_pow, [float, float], float, name="pow")
        self.pset.addPrimitive(np.sign, [float], float, name="sign")
        self.pset.addPrimitive(tanh_scaled, [float], float, name="tanh")

        # 条件运算
        def if_then_else(condition, output1, output2):
            return output1 if condition > 0 else output2

        self.pset.addPrimitive(if_then_else, [float, float, float], float, name="ite")

        # 常数
        self.pset.addEphemeralConstant(
            "rand_const", lambda: random.uniform(-2.0, 2.0), float
        )
        self.pset.addTerminal(0.0, float, name="zero")
        self.pset.addTerminal(1.0, float, name="one")

        # 重命名参数
        if self.feature_names is not None:
            for i, name in enumerate(self.feature_names):
                self.pset.renameArguments(**{f"ARG{i}": name})

    def _evaluate_multi_objective(self, individual) -> Tuple[float, float]:
        """
        多目标评估函数
        目标1: 最小化峰度偏差
        目标2: 最小化复杂度
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
                return (1000.0, 1000.0)

        # 检查有效性
        if len(y_pred) <= 2 or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return (1000.0, 1000.0)

        # 计算峰度目标
        try:
            cumsum_threshold = np.sum(np.abs(y_pred)) / (
                len(self.candles_in_metrics) // 120
            )

            if cumsum_threshold <= 0:
                return (1000.0, 1000.0)

            merged_bar = build_bar_by_cumsum(
                self.candles_in_metrics,
                np.abs(y_pred),
                cumsum_threshold,
                reverse=False,
            )

            if len(merged_bar) < len(self.candles_in_metrics) // 240:
                return (1000.0, 1000.0)

            kurtosis_deviation = self._calculate_kurtosis(merged_bar)

        except Exception:
            return (1000.0, 1000.0)

        # 计算复杂度（考虑深度和大小）
        complexity = len(individual) + individual.height * 2

        return (kurtosis_deviation, complexity)

    def _calculate_kurtosis(self, merged_bar: np.ndarray) -> float:
        """计算峰度偏差"""
        close_arr = merged_bar[:, 2]

        if len(close_arr) < 10:
            return 1000.0

        ret = np.log(close_arr[5:] / close_arr[:-5])
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]

        if len(ret) < 10 or ret.std() < 1e-10:
            return 1000.0

        standard = (ret - ret.mean()) / ret.std()
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        # 同时考虑峰度和偏度
        skewness = abs(stats.skew(standard))

        return abs(kurtosis - 3.0) + 0.1 * skewness

    def _semantic_crossover(self, ind1, ind2):
        """
        基于语义相似度的交叉操作
        """
        # 计算两个个体的语义输出
        func1 = gp.compile(expr=ind1, pset=self.pset)
        func2 = gp.compile(expr=ind2, pset=self.pset)

        try:
            output1 = np.array([func1(*x) for x in self.X_scaled[:100]])  # 使用子集加速
            output2 = np.array([func2(*x) for x in self.X_scaled[:100]])

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

    def _island_evolution(
        self,
        island_id: int,
        population: List,
        stats: tools.Statistics,
        generations: int,
    ) -> Tuple[List, tools.Logbook]:
        """
        岛屿进化
        """
        logbook = tools.Logbook()
        logbook.header = ["gen", "island", "nevals"] + stats.fields

        # 评估初始种群
        fitnesses = list(map(self.toolbox.evaluate, population))
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
            offspring = self.toolbox.select(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # 交叉和变异
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = self.toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            for i in range(len(offspring)):
                if random.random() < current_mutation_prob:
                    if random.random() < 0.5:
                        (offspring[i],) = self.toolbox.mutate(offspring[i])
                    else:
                        (offspring[i],) = self.toolbox.mutate_eph(offspring[i])
                    del offspring[i].fitness.values

                # 局部搜索
                if random.random() < self.local_search_prob:
                    (offspring[i],) = self.toolbox.local_search(offspring[i])
                    del offspring[i].fitness.values

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 环境选择
            population[:] = self.toolbox.select(population + offspring, len(population))

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

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        candles_path: str = "data/btc_1m.npy",
    ):
        """训练模型"""
        self.X = X
        self.feature_names = feature_names

        # 特征标准化
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # 加载数据
        self.candles = np.load(candles_path)
        self.candles = self.candles[self.candles[:, 5] > 0]

        if len(self.candles) > 10000:
            test_start_idx = len(self.candles) - 10000
            self.candles = self.candles[test_start_idx:]
            self.X = self.X[-10000:] if len(self.X) > 10000 else self.X
            self.X_scaled = (
                self.X_scaled[-10000:] if len(self.X_scaled) > 10000 else self.X_scaled
            )

        self.NA_MAX_NUM = 0
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

        # 创建岛屿种群
        islands = []
        for i in range(self.n_islands):
            pop = self.toolbox.population(n=self.population_size // self.n_islands)
            islands.append(pop)

        # 主进化循环
        all_logbooks = []

        for gen in range(0, self.generations, 10):  # 每10代进行一次迁移
            # TODO: 需要改为多进程执行。由于DEAP creator的序列化问题，暂时使用单线程执行
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
        """可视化Pareto前沿"""
        if self.pareto_front is None:
            raise ValueError("模型尚未训练")

        plt.figure(figsize=(10, 6))

        # 提取目标值
        objectives = np.array([ind.fitness.values for ind in self.pareto_front])

        plt.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=50)
        plt.xlabel("峰度偏差 (Kurtosis Deviation)")
        plt.ylabel("复杂度 (Complexity)")
        plt.title("Pareto Front - 峰度 vs 复杂度")
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
        """获取最佳的n个表达式及其性能"""
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
            cumsum_threshold = np.sum(np.abs(y_pred)) / (
                len(self.candles_in_metrics) // 120
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
        """预测"""
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
        """保存模型"""
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
        """加载模型"""
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
    """准备特征数据"""
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
