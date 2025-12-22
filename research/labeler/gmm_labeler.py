import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hmmlearn.hmm import GMMHMM
from jesse.helpers import timestamp_to_time
from plotly.subplots import make_subplots


def gmm_labeler_find_best_params(
    candles: np.ndarray, lag_n: int, verbose: bool = True
) -> dict:
    def objective(trial: optuna.Trial):
        close_arr = candles[:, 2]
        high_arr = candles[:, 3][lag_n:]
        low_arr = candles[:, 4][lag_n:]

        log_return = np.log(close_arr[1:] / close_arr[:-1])[lag_n - 1 :]
        log_return_L = np.log(close_arr[lag_n:] / close_arr[:-lag_n])
        HL_diff = np.log(high_arr / low_arr)

        X = np.column_stack([log_return_L, log_return, HL_diff])

        datelist = np.asarray(
            [pd.Timestamp(timestamp_to_time(i)) for i in candles[:, 0][lag_n:]]
        )
        closeidx = candles[:, 2][lag_n:]

        assert len(datelist) == len(closeidx)
        assert len(datelist) == len(X)

        try:
            gmm = GMMHMM(
                n_components=2,
                n_mix=3,
                covariance_type="diag",
                n_iter=369,
                random_state=trial.suggest_int("random_state", 0, 1000),
            )
            gmm.fit(X)
            latent_states_sequence = gmm.predict(X)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            # GMM拟合或预测失败时，返回一个无效值让Optuna忽略这个trial
            # 使用负无穷或NaN会让Optuna认为这个trial失败
            if verbose:
                print(f"GMM failed with error: {e}")
            return float("-inf")  # 返回负无穷，因为我们在maximize

        data = pd.DataFrame(
            {
                "datelist": datelist,
                "logreturn": log_return,
                "state": latent_states_sequence,
            }
        ).set_index("datelist")

        final_ret = 0
        for i in data["state"].unique():
            ret = data[data["state"] == i]["logreturn"].sum()
            final_ret += np.abs(ret)

        return final_ret

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=25)

    # 检查是否有有效的trial
    if (
        len(
            [
                t
                for t in study.trials
                if t.value is not None and t.value != float("-inf")
            ]
        )
        == 0
    ):
        # 如果所有trial都失败了，返回一个默认的random_state
        if verbose:
            print("Warning: All GMM trials failed, using default random_state=42")
        return {"random_state": 369}

    best_params = study.best_params

    # 🔧 显式清理 Optuna study，防止内存泄漏
    del study
    gc.collect()

    return best_params


class GMMLabeler:
    def __init__(self, candles: np.ndarray, lag_n: int, verbose: bool = True):
        self.lag_n = lag_n

        random_state = gmm_labeler_find_best_params(candles, lag_n, verbose)[
            "random_state"
        ]
        self.gmm_model = GMMHMM(
            n_components=2,
            n_mix=3,
            covariance_type="diag",
            n_iter=369,
            random_state=random_state,
        )

        close_arr = candles[:, 2]
        high_arr = candles[:, 3][lag_n:]
        low_arr = candles[:, 4][lag_n:]

        self._datelist = np.asarray(
            [pd.Timestamp(timestamp_to_time(i)) for i in candles[:, 0][lag_n:]]
        )
        self._closeidx = candles[:, 2][lag_n:]

        self.log_return = np.log(close_arr[1:] / close_arr[:-1])[lag_n - 1 :]
        log_return_L = np.log(close_arr[lag_n:] / close_arr[:-lag_n])
        HL_diff = np.log(high_arr / low_arr)
        X = np.column_stack([log_return_L, self.log_return, HL_diff])

        self.gmm_model.fit(X)
        self.latent_states_sequence = self.gmm_model.predict(X)  ### 硬标签
        self.state_probabilities = self.gmm_model.predict_proba(X)  ### 概率标签

        # Label_b: assign each return to the previous state's label.
        state_0_mask = np.append(0, (self.latent_states_sequence == 0)[:-1])
        state_1_mask = np.append(0, (self.latent_states_sequence == 1)[:-1])
        state_0_return = (self.log_return * state_0_mask).sum()
        state_1_return = (self.log_return * state_1_mask).sum()
        if state_0_return > state_1_return:
            self.latent_states_sequence = 1 - self.latent_states_sequence
            self.state_probabilities = self.state_probabilities[:, ::-1]
        self.buy_state = 1

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

        for i in range(self.gmm_model.n_components):
            state = self.latent_states_sequence == i
            idx = np.append(0, state[:-1])
            ret = data.logreturn.multiply(idx, axis=0).sum()
            count = int(idx.sum())
            print(f"state {i} ({count}) return: {ret:.6%}")

        plt.figure(figsize=(20, 8))
        for i in range(self.gmm_model.n_components):
            state = self.latent_states_sequence == i
            idx = np.append(0, state[:-1])
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
