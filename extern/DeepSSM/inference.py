########### 实时特征输出 ###########

import numpy as np
import torch


#### 模型定义


class DeepSSM(torch.nn.Module):
    def __init__(self, obs_dim, state_dim=5, lstm_hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.lstm = torch.nn.LSTM(obs_dim, lstm_hidden, batch_first=True, num_layers=1)
        self.transition = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden + state_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 2 * state_dim),
        )
        self.observation = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 2 * obs_dim),
        )

    def get_transition_dist(self, lstm_out, z_prev):
        input_ = torch.cat([lstm_out, z_prev], dim=-1)
        out = self.transition(input_)
        mean, log_var = torch.split(out, out.size(-1) // 2, dim=-1)
        return mean, torch.clamp(log_var, -10, 10)

    def get_observation_dist(self, z):
        out = self.observation(z)
        mean, log_var = torch.split(out, out.size(-1) // 2, dim=-1)
        return mean, torch.clamp(log_var, -10, 10)


######## 雅可比矩阵


def compute_jacobian_numerical(f, x, eps=1e-6):

    x = x.detach().clone()
    y = f(x)
    obs_dim, state_dim = y.shape[1], x.shape[1]
    jac = torch.zeros(obs_dim, state_dim)
    for i in range(state_dim):
        x_eps = x.clone()
        x_eps[0, i] += eps
        jac[:, i] = (f(x_eps) - y).squeeze(0) / eps
    return jac


##### 实时特征生成
class DeepSSMRealTime:
    def __init__(self, model_path):
        ##### 加载模型和标准化
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.scaler = checkpoint["scaler"]
        self.model = DeepSSM(
            obs_dim=checkpoint["obs_dim"],
            state_dim=checkpoint["state_dim"],
            lstm_hidden=checkpoint["lstm_hidden"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  ###### 评估模式

        ###### 初始化状态变量
        self.state_dim = checkpoint["state_dim"]
        self.obs_dim = checkpoint["obs_dim"]
        self.z = torch.zeros(1, self.state_dim)  ###### 状态估计
        self.P = torch.eye(self.state_dim) * 0.1  ###### 协方差矩阵
        self.lstm_hidden = torch.zeros(
            1, 1, self.model.lstm.hidden_size
        )  ###### LSTM状态
        self.lstm_cell = torch.zeros(1, 1, self.model.lstm.hidden_size)  ###### LSTM状态

    def process(self, new_data):

        ##### 标准化新数据
        new_data_scaled = self.scaler.transform(new_data.reshape(1, -1))
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        ##### LSTM单步更新
        lstm_out, (self.lstm_hidden, self.lstm_cell) = self.model.lstm(
            new_data_tensor.unsqueeze(0),  ###### 形状：[1,1,obs_dim]
            (self.lstm_hidden, self.lstm_cell),
        )
        lstm_out = lstm_out.squeeze(0)  ###### 形状：[1, lstm_hidden]

        ##### 扩展卡尔曼滤波状态预测
        with torch.no_grad():
            trans_mean, trans_logvar = self.model.get_transition_dist(lstm_out, self.z)
            z_pred = trans_mean  ##### 预测状态
            trans_var = torch.diag(torch.exp(trans_logvar.squeeze(0)))
            P_pred = trans_var  ##### 预测协方差

        #### 观测模型与雅可比矩阵
        with torch.no_grad():
            obs_mean, obs_logvar = self.model.get_observation_dist(z_pred)

        #### 计算雅可比矩阵
        H = compute_jacobian_numerical(
            lambda x: self.model.get_observation_dist(x)[0], z_pred
        )

        #### 卡尔曼增益与状态更新
        obs_var = torch.exp(obs_logvar.squeeze(0))
        R = (
            torch.diag(obs_var)
            if len(obs_var) == self.obs_dim
            else torch.eye(self.obs_dim) * 0.1
        )
        H_t = H.T
        temp = H @ P_pred @ H_t + R + torch.eye(self.obs_dim) * 1e-6  ##### 数值稳定
        K = P_pred @ H_t @ torch.inverse(temp)  #####卡尔曼增益

        #####更新状态和协方差
        error = (new_data_tensor - obs_mean).T  ##### 残差
        self.z = (z_pred.T + K @ error).T  ##### 新状态估计
        self.P = (torch.eye(self.state_dim) - K @ H) @ P_pred  ##### 新协方差

        ##### 返回生成的特征
        return self.z.detach().numpy().flatten()


if __name__ == "__main__":

    ##### 导入模型
    realtime = DeepSSMRealTime("deep_ssm_model.pt")
    print(
        f"实时特征生成器初始化完成（{realtime.obs_dim}维输入，{realtime.state_dim}维特征）"
    )

    ##### 要用实际数据源
    for i in range(5):
        ##### 生成模拟数据
        new_data = np.random.randn(realtime.obs_dim)

        ##### 生成特征
        feature = realtime.process(new_data)

        ##### 实时推理需要结果给到下游模型
        print(f"第{i+1}条数据特征：{feature[:3].round(4)}...")
