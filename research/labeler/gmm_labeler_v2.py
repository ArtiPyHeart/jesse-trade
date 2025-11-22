import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hmmlearn.hmm import GMMHMM
from jesse.helpers import timestamp_to_time
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler


class GMMLabelerV2:
    def __init__(self, candles: np.ndarray, lag_n: int = 5, verbose: bool = False):
        self.lag_n = lag_n

        # Extract candle data
        timestamps = candles[:, 0]
        closes = candles[:, 2]
        highs = candles[:, 3]
        lows = candles[:, 4]

        # --- Feature Engineering ---
        # 针对自定义K线（已有去噪特性，非等时），我们精简特征，避免过度工程化
        # 只保留核心的价格行为特征和用于Labeling的未来特征

        s_close = pd.Series(closes)
        s_high = pd.Series(highs)
        s_low = pd.Series(lows)

        # 1. 基础特征 (保留用户原有的核心逻辑)
        # log_return: 当前Bar的收益率 (描述即时动量)
        ret_1 = np.log(s_close / s_close.shift(1))
        # log_return_L: 过去Lag_N个Bar的收益率 (描述短期趋势)
        ret_lag = np.log(s_close / s_close.shift(lag_n))
        # HL_diff: 波动幅度/K线形态 (描述当前Bar的结构)
        hl_diff = np.log(s_high / s_low)

        # 2. Oracle Feature (未来特征)
        # 用于训练时的Labeling，这是最关键的改进。
        # 我们引入未来Lag_N的收益率，帮助HMM在聚类时能利用"未来发生了什么"来定义当前的状态。
        # 例如：如果未来5根K线大涨，当前这根K线应该属于"Bull"状态。
        fwd_ret_lag = np.log(s_close.shift(-lag_n) / s_close)

        # 组装特征
        df_features = pd.DataFrame(
            {
                "ret_1": ret_1,
                "ret_lag": ret_lag,
                "hl_diff": hl_diff,
                "fwd_ret_lag": fwd_ret_lag,
                "timestamp": timestamps,
                "close": closes,
            }
        )

        df_features = df_features.dropna()

        if len(df_features) < 100:
            if verbose:
                print("Warning: Not enough data after feature engineering.")

        # --- 特征选择与处理 ---
        # 我们将使用以下特征进行聚类：
        # 1. hl_diff: 形态
        # 2. ret_lag: 过去趋势
        # 3. ret_1: 当前动量
        # 4. fwd_ret_lag: 未来趋势 (这是Labeling的核心“作弊”特征)
        feature_cols = ["hl_diff", "ret_lag", "ret_1", "fwd_ret_lag"]
        X_raw = df_features[feature_cols].values

        # 标准化：GMMHMM是基于距离的算法，必须标准化，否则量级大的特征(如hl_diff可能较小)会被忽略
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_raw)

        # 存储用于绘图和属性的数据
        self._datelist = np.asarray(
            [
                pd.Timestamp(timestamp_to_time(t))
                for t in df_features["timestamp"].values
            ]
        )
        self._closeidx = df_features["close"].values
        self.log_return = df_features["ret_1"].values

        # --- 改进点：全协方差矩阵 (Full Covariance) ---
        # 金融数据中，波动率(HL_diff)和收益率(Returns)往往存在相关性（例如大波动往往伴随大收益或反转）。
        # 'diag' 假设特征独立，会丢失这种相关性信息。'full' 能更好地拟合这种复杂的联合分布。
        covariance_type = "full"

        # --- 改进点：方向感知初始化 (Direction-Aware Initialization) ---
        # 不使用随机种子，而是利用未来收益率 fwd_ret_lag 显式地初始化两个状态。
        # 这样可以保证 State 0 和 State 1 有明确的物理意义（空/多），而不是随机的。

        fwd_ret_idx = feature_cols.index("fwd_ret_lag")
        # 简单切分：未来涨就是多头初始集，未来跌就是空头初始集
        up_mask = X[:, fwd_ret_idx] > 0
        down_mask = ~up_mask

        X_up = X[up_mask]
        X_down = X[down_mask]

        n_components = 2
        n_mix = 3  # 保持3个混合分量，足以覆盖"主趋势"、"震荡"、"极端行情"三种微观结构
        n_features = X.shape[1]

        # 初始化参数容器
        means_init = np.zeros((n_components, n_mix, n_features))
        # full covariance shape: (n_components, n_mix, n_features, n_features)
        covars_init = np.zeros((n_components, n_mix, n_features, n_features))
        weights_init = np.full((n_components, n_mix), 1.0 / n_mix)

        # 计算初始统计量
        # 这里我们假设每个状态内部的3个mix初始时是相似的，让EM算法去分化它们
        if len(X_down) > 10:
            d_mean = X_down.mean(axis=0)
            d_cov = np.cov(X_down.T) + np.eye(n_features) * 1e-6
        else:
            d_mean = np.zeros(n_features)
            d_cov = np.eye(n_features)

        if len(X_up) > 10:
            u_mean = X_up.mean(axis=0)
            u_cov = np.cov(X_up.T) + np.eye(n_features) * 1e-6
        else:
            u_mean = np.zeros(n_features)
            u_cov = np.eye(n_features)

        # 填充初始化矩阵
        # Index 0 -> Down (基于 mask 取反)
        # Index 1 -> Up
        for m in range(n_mix):
            means_init[0, m] = d_mean
            covars_init[0, m] = d_cov
            means_init[1, m] = u_mean
            covars_init[1, m] = u_cov

        # 强粘性转移矩阵初始化
        # 自定义K线已经去噪，趋势性更强，因此我们期望状态保持(Stay probability)较高
        transmat_init = np.array([[0.90, 0.10], [0.10, 0.90]])
        startprob_init = np.array([0.5, 0.5])

        self.gmm_model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=200,
            tol=1e-3,
            verbose=verbose,
            init_params="",  # 禁用自动初始化
            random_state=42,
        )

        self.gmm_model.startprob_ = startprob_init
        self.gmm_model.transmat_ = transmat_init
        self.gmm_model.means_ = means_init
        self.gmm_model.covars_ = covars_init
        self.gmm_model.weights_ = weights_init

        if verbose:
            print("Fitting GMMHMM with Custom Candles strategy...")
            print(f"Features: {feature_cols}")

        try:
            self.gmm_model.fit(X)
        except Exception as e:
            if verbose:
                print(f"Fit failed: {e}. Fallback to default init.")
            self.gmm_model.init_params = "stmcw"
            self.gmm_model.fit(X)

        self.latent_states_sequence = self.gmm_model.predict(X)
        self.state_probabilities = self.gmm_model.predict_proba(X)

        # --- Post-Processing: 简单的平滑 ---
        # 自定义K线虽然去噪，但在趋势转换点仍可能有抖动。
        # 我们使用极短窗口的平滑来过滤单点噪音。
        probs_df = pd.DataFrame(self.state_probabilities)
        self.state_probabilities = (
            probs_df.rolling(3, center=True, min_periods=1).mean().values
        )
        self.latent_states_sequence = np.argmax(self.state_probabilities, axis=1)

        # 确定 Buy State (通常 Index 1 是 Up，但为了保险再次校验)
        state_0_ret = self.log_return[self.latent_states_sequence == 0].mean()
        state_1_ret = self.log_return[self.latent_states_sequence == 1].mean()

        if state_1_ret > state_0_ret:
            self.buy_state = 1
        else:
            self.buy_state = 0

        if verbose:
            print(f"State 0 Avg Ret: {state_0_ret:.6%}")
            print(f"State 1 Avg Ret: {state_1_ret:.6%}")
            print(f"Buy State: {self.buy_state}")

    def plot_label_on_candles(self):
        """
        在蜡烛图上绘制标签
        """
        n_states = self.gmm_model.n_components
        # 主图 + 每个隐含状态一个概率副图
        prob_heights = [0.4 / n_states] * n_states
        row_heights = [0.6] + prob_heights
        subplot_titles = ["隐含状态序列"] + [f"P(state={i})" for i in range(n_states)]

        fig = make_subplots(
            rows=1 + n_states,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.06,
            subplot_titles=tuple(subplot_titles),
        )
        colors = px.colors.qualitative.Plotly

        for i in range(n_states):
            state = self.latent_states_sequence == i
            fig.add_trace(
                go.Scatter(
                    x=self._datelist[state],
                    y=self._closeidx[state],
                    mode="markers",
                    name=f"latent state {i}",
                    marker=dict(color=colors[i % len(colors)], size=4),
                    legendgroup=f"state_{i}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # 概率副图：为每个隐含状态单独绘制概率线图或面积图
        for i in range(n_states):
            fig.add_trace(
                go.Scatter(
                    x=self._datelist,
                    y=self.state_probabilities[:, i],
                    name=f"P(state={i})",
                    mode="lines",
                    fill="tozeroy",  # 填充到0，形成面积图
                    line=dict(color=colors[i % len(colors)], width=1),
                    fillcolor=colors[i % len(colors)],
                    opacity=0.7,
                    legendgroup=f"state_{i}",
                    showlegend=False,  # 与主图同组颜色，不重复图例
                ),
                row=2 + i,
                col=1,
            )

        fig.update_yaxes(title_text="收盘价", row=1, col=1)
        for i in range(n_states):
            fig.update_yaxes(title_text=f"P(state={i})", range=[0, 1], row=2 + i, col=1)
        # 仅在最底部子图设置时间轴标题
        fig.update_xaxes(title_text="时间", row=1 + n_states, col=1)

        fig.update_layout(
            title="隐含状态序列与状态概率",
            showlegend=True,
            hovermode="x unified",  # 统一的悬停模式，更好地显示多个子图的数据
        )

        fig.show()

    def plot_label_returns(self):
        """
        绘制标签收益
        """
        data = pd.DataFrame(
            {
                "datelist": self._datelist,
                "logreturn": self.log_return,
                "state": self.latent_states_sequence,
            }
        ).set_index("datelist")

        for i in data["state"].unique():
            ret = data[data["state"] == i]["logreturn"].sum()
            count = data[data["state"] == i].shape[0]
            print(f"state {i} ({count}) return: {ret:.6%}")

        plt.figure(figsize=(20, 8))
        for i in range(self.gmm_model.n_components):
            state = self.latent_states_sequence == i
            idx = np.append(0, state[1:])
            data[f"state {i}_return"] = data.logreturn.multiply(idx, axis=0)
            plt.plot(
                np.exp(data[f"state {i}_return"].cumsum()),
                label=f"latent_state {i}",
            )
            plt.legend(loc="upper left")
            plt.grid(1)

        plt.show()

    @property
    def label_hard_state(self):
        """
        多空二分类硬标签，用于分类模型
        """
        return (self.latent_states_sequence == self.buy_state).astype(int)

    @property
    def label_double_prob(self):
        """
        双概率标签，完整概率分布，用于多标签回归
        """
        buy_prob = self.state_probabilities[:, self.buy_state]
        sell_prob = self.state_probabilities[:, 1 - self.buy_state]
        return np.column_stack([sell_prob, buy_prob])

    @property
    def label_directional_prob(self):
        """
        带方向的原始概率标签，方向 + 概率大小（非对称），用于回归
        """
        return np.where(
            self.latent_states_sequence == self.buy_state,
            self.state_probabilities[:, self.buy_state],
            -self.state_probabilities[:, 1 - self.buy_state],
        )

    @property
    def label_direction_force(self):
        """
        方向强度标签，方向 + 相对强度（对称），用于回归
        """
        return self.state_probabilities[:, self.buy_state] * 2 - 1
