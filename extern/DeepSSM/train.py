import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from sklearn.preprocessing import StandardScaler


##### 加载数据


def load_data(csv_path, device):
    df = pd.read_csv(csv_path)
    features = df.values
    #### 如果是差分处理过的数据可以不用做标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return torch.tensor(features_scaled, dtype=torch.float32, device=device), scaler


##### SSM模型


class DeepSSM(nn.Module):

    ### 结合LSTM数据的时间依赖，用参数化状态转移和观测模型来实现数据的状态推断

    def __init__(self, obs_dim, state_dim=5, lstm_hidden=64):

        super().__init__()
        self.obs_dim = obs_dim  ######  观测数据的维度，（原始数据）
        self.state_dim = state_dim  #####  潜在状态的维度，即输出状态特征的维度
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,  ####LSTM隐藏层的维度，用于提取序列特征
            batch_first=True,
            num_layers=1,  ####### 单层 LSTM
        )
        self.transition = nn.Sequential(
            nn.Linear(lstm_hidden + state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2 * state_dim),
        )
        self.observation = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 2 * obs_dim)
        )
        self.initial_state_mean = nn.Parameter(torch.zeros(state_dim))
        self.initial_state_log_var = nn.Parameter(torch.zeros(state_dim))

    #### 计算状态转移分布的参数（均值和对数方差）

    def get_transition_dist(self, lstm_out, z_prev):
        input_ = torch.cat([lstm_out, z_prev], dim=-1)
        out = self.transition(input_)
        mean, log_var = torch.split(out, out.size(-1) // 2, dim=-1)
        return mean, torch.clamp(log_var, -10, 10)

    #### 计算观测分布的参数（均值和对数方差）

    def get_observation_dist(self, z):
        out = self.observation(z)
        mean, log_var = torch.split(out, out.size(-1) // 2, dim=-1)
        return mean, torch.clamp(log_var, -10, 10)


###### 数值法计算雅可比矩阵,用于扩展卡尔曼的线性化


def compute_jacobian_numerical(f, x, eps=1e-6):
    x = x.detach().clone()
    y = f(x)
    obs_dim = y.shape[1]
    state_dim = x.shape[1]
    jac = torch.zeros(obs_dim, state_dim, device=x.device)
    for i in range(state_dim):
        x_eps = x.clone()
        x_eps[0, i] += eps
        y_eps = f(x_eps)
        jac[:, i] = (y_eps - y).squeeze(0) / eps
    return jac


##### 嵌入LSTM的概率模型定义


def deep_ssm_model(y, model):

    ###### 基于Pyro概率模型定义状态空间模型的生成过程，（先验和观测模型）
    #####  从初始状态按时间步转移状态，再由状态生成观测数据

    batch_size, T, obs_dim = y.shape
    state_dim = model.state_dim

    ###### 用LSTM提取整个观测序列的特征
    lstm_out, _ = model.lstm(y)

    #####初始状态的先验分布
    z0_mean = model.initial_state_mean.expand(batch_size, -1)
    z0_log_var = model.initial_state_log_var.expand(batch_size, -1)
    z = pyro.sample("z0", dist.Normal(z0_mean, torch.exp(0.5 * z0_log_var)).to_event(1))

    ### 按时间步迭代，定义状态转移和观测过程
    for t in range(1, T):
        transition_mean, transition_log_var = model.get_transition_dist(
            lstm_out[:, t, :], z
        )
        z = pyro.sample(
            f"z{t}",
            dist.Normal(transition_mean, torch.exp(0.5 * transition_log_var)).to_event(
                1
            ),
        )
        obs_mean, obs_log_var = model.get_observation_dist(z)
        pyro.sample(
            f"y{t}",
            dist.Normal(obs_mean, torch.exp(0.5 * obs_log_var)).to_event(1),
            obs=y[:, t, :],
        )


def deep_ssm_guide(y, model):

    ###### 定义变分分布（近似后验分布）用于变分推断
    ####### 为了计算和优化，所以会用参数化的分布近似状态的后验分布

    batch_size, T, obs_dim = y.shape
    state_dim = model.state_dim

    ###### 定义状态后验的均值（z_loc）和标准差（z_scale）参数

    z_loc = pyro.param("z_loc", torch.zeros(batch_size, T, state_dim, device=y.device))
    z_scale = pyro.param(
        "z_scale",
        torch.ones(batch_size, T, state_dim, device=y.device) * 0.1,
        constraint=dist.constraints.positive,
    )

    ###### 按时间步定义每个状态zt的变分分布

    for t in range(T):
        pyro.sample(f"z{t}", dist.Normal(z_loc[:, t, :], z_scale[:, t, :]).to_event(1))


######### 使用扩展卡尔曼滤波
#######  在已知观测序列的情况下递归估计潜在状态


def deep_ssm_kalman_filter(y_seq, model):
    T = len(y_seq)
    state_dim = model.state_dim
    obs_dim = model.obs_dim
    device = y_seq.device

    lstm_hidden = torch.zeros(1, 1, model.lstm.hidden_size, device=device)
    lstm_cell = torch.zeros(1, 1, model.lstm.hidden_size, device=device)
    z = model.initial_state_mean.unsqueeze(0).to(device)
    P = torch.diag(torch.exp(model.initial_state_log_var)).to(device)

    states = [z.squeeze(0).cpu()]

    ### 按时间步进行
    for t in range(1, T):

        ###### 用LSTM处理当前观测来获取特征

        y_t = y_seq[t].unsqueeze(0).unsqueeze(0)
        lstm_out, (lstm_hidden, lstm_cell) = model.lstm(y_t, (lstm_hidden, lstm_cell))
        lstm_out = lstm_out.squeeze(0)

        #### 基于前一个状态和LSTM特征来计算当前状态的先验分布

        transition_mean, transition_log_var = model.get_transition_dist(lstm_out, z)
        z_pred = transition_mean
        transition_var = torch.diag(torch.exp(transition_log_var.squeeze(0)))
        P_pred = transition_var.to(device)

        y_t_obs = y_seq[t].unsqueeze(0)
        obs_mean, obs_log_var = model.get_observation_dist(z_pred)

        ###### 定义观测函数（输入状态，输出观测均值）用于计算雅可比矩阵

        def observation_func(x):
            return model.get_observation_dist(x)[0]

        ######  用数值计算观测函数在预测状态的雅可比矩阵

        H = compute_jacobian_numerical(observation_func, z_pred)
        H = H[:obs_dim, :state_dim].to(device)

        #######  观测噪声协方差矩阵

        obs_var = torch.exp(obs_log_var.squeeze(0)).to(device)
        R = (
            torch.diag(obs_var)
            if len(obs_var) == obs_dim
            else torch.eye(obs_dim, device=device) * 0.1
        )

        #### 计算卡尔曼增益，用于权衡预测和观测的可信度

        H_t = H.T
        temp = H @ P_pred @ H_t + R
        temp_inv = torch.inverse(temp + torch.eye(obs_dim, device=device) * 1e-6)
        K = P_pred @ H_t @ temp_inv

        ##### 用观测残差更新状态

        error = (y_t_obs - obs_mean).T.to(device)
        z = (z_pred.T + K @ error).T
        P = (torch.eye(state_dim, device=device) - K @ H) @ P_pred

        states.append(z.squeeze(0).cpu())

    return torch.stack(states), P.cpu()


##### 模型保存和加载


def save_deep_model(model, scaler, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "state_dim": model.state_dim,
            "lstm_hidden": model.lstm.hidden_size,
            "scaler": scaler,
        },
        path,
    )
    print(f"模型已保存到 {path}")


def load_deep_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = DeepSSM(
        obs_dim=checkpoint["obs_dim"],
        state_dim=checkpoint["state_dim"],
        lstm_hidden=checkpoint["lstm_hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["scaler"]


###### 参数配置


def main():
    config = {
        "csv_path": "./np_fracdiff_features.csv",  ######### 原始特征文件
        "model_path": "deep_ssm_model.pt",  ######### 模型保存
        "feature_save_path": "deep_ssm_features.csv",  ######### 特征保存
        "state_dim": 5,  ######### 新特征数据维度数量
        "lstm_hidden": 64,  ######### LSTM 隐藏层维度
        "max_epochs": 50,  ###### 最大训练轮数
        "patience": 5,  ###### 早停耐心值，5个epoch无改善则停止
        "min_delta": 0.01,  ###### 损失改善的最小阈值
        "lr": 0.001,  ###### 学习率
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    ##### 加载数据
    y_raw, scaler = load_data(config["csv_path"], device)
    T, obs_dim = y_raw.shape
    y = y_raw.unsqueeze(0)
    print(f"数据加载完成：{T}行，{obs_dim}维特征，批次形状：{y.shape}")

    ##### 初始化模型
    model = DeepSSM(
        obs_dim=obs_dim,
        state_dim=config["state_dim"],
        lstm_hidden=config["lstm_hidden"],
    ).to(device)

    ##### 训练模型
    optimizer = pyro.optim.Adam({"lr": config["lr"]})
    svi = SVI(
        model=lambda: deep_ssm_model(y, model),
        guide=lambda: deep_ssm_guide(y, model),
        optim=optimizer,
        loss=Trace_ELBO(),
    )
    print("开始训练DeepSSM模型...")

    ##### 早停参数初始化
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["max_epochs"]):
        loss = svi.step()
        avg_loss = loss / T  ##### 平均损失

        ##### 打印每10个epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['max_epochs']} | Loss: {avg_loss:.4f}")

        ##### 早停逻辑
        if avg_loss < best_loss - config["min_delta"]:
            best_loss = avg_loss
            patience_counter = 0  ##### 重置计数器
        else:
            patience_counter += 1
            ##### 达到耐心值，停止训练
            if patience_counter >= config["patience"]:
                print(f"早停触发：第{epoch+1}轮损失未改善，停止训练")
                break

    ##### 生成特征
    states, _ = deep_ssm_kalman_filter(y_raw, model)
    print(
        f"特征生成完成，形状：{states.shape}（{states.shape[0]}行，{states.shape[1]}维）"
    )

    ##### 保存特征
    feature_df = pd.DataFrame(
        states.detach().numpy(),
        columns=[f"deep_ssm_feature_{i}" for i in range(config["state_dim"])],
    )
    feature_df.to_csv(config["feature_save_path"], index=False)
    print(f"特征已保存到 {config['feature_save_path']}")

    ##### 保存模型
    save_deep_model(model, scaler, config["model_path"])
    return states


if __name__ == "__main__":
    states = main()
