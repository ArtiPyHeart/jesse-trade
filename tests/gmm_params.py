import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GMMHMM
from jesse.helpers import timestamp_to_time

merged_bar = np.load(
    "data/merged_bar.npy"
)  # 6 * N的K线，timestamp, open, close, high, low, volume

L = 5  ### X阶
mix = 3  ### GMM mix参数

close_arr = merged_bar[:, 2]
high_arr = merged_bar[:, 3][L:]
low_arr = merged_bar[:, 4][L:]

# lag 1的log return，比正态分布峰度更高
log_return = np.log(close_arr[1:] / close_arr[:-1])[L - 1 :]
# lag X的log return，可以通过调整X阶的大小调整与正态分布的接近程度，通常X阶越大，峰度越低，越接近正态分布
log_return_L = np.log(close_arr[L:] / close_arr[:-L])
HL_diff = np.log(high_arr / low_arr)

X = np.column_stack([HL_diff, log_return_L, log_return])

datelist = np.asarray(
    [pd.Timestamp(timestamp_to_time(i)) for i in merged_bar[:, 0][L:]]
)
closeidx = merged_bar[:, 2][L:]

assert len(datelist) == len(closeidx)
assert len(datelist) == len(X)

# ======== 1. 结构相关 ========
n_components = 2  # 隐状态数：2 ⇒ 上涨 vs 下跌
n_mix = 3  # 混合高斯个数，↑ 能拟合厚尾/多峰，过大易过拟合
cov_type = "diag"  # {"diag","full","tied","spherical"}
#  full: 最灵活，样本少时易病态；diag: 常用且稳健

# ======== 2. 数值稳定 / 收敛控制 ========
min_covar = 1e-3  # 对角线加噪，防止协方差奇异；如果报奇异或拟合抖动，可↑
tol = 1e-3  # 对数似然增量阈值；想更充分收敛可↓
n_iter = 1000  # EM 最大迭代；若常提前收敛，可↓tol 或 ↑n_iter
implementation = "log"  # {"log","scaling"}，log 更稳定

# ======== 3. 先验（提高鲁棒性，缓解过拟合）======
# Dirichlet 先验：>1 ⇒ 平滑（不易把概率压到 0），<1 ⇒ 稀疏化
startprob_prior = np.array([5.0, 5.0])  # 初始状态分布先验
transmat_prior = np.full((2, 2), 5.0)  # 转移矩阵先验

weights_prior = 2.0  # GMM 权重 Dirichlet α
means_prior = 0.0  # 高斯均值 μ₀，可设为样本均值
means_weight = 0.5  # “等效样本数”λ，>0 会把均值拉回 μ₀
# 协方差先验：根据 cov_type 自动解释
covars_prior = None  # =None 时库里会给弱信息缺省
covars_weight = None

# ======== 4. 其它可控项 ========
algorithm = "viterbi"  # {"viterbi", "map"}；map 平滑、对噪声略稳
params = "stmcw"  # 哪些参数在 M 步更新；可冻结某些字母抑制过拟合
init_params = "stmcw"  # 首次 fit() 时哪些参数需要自动初始化
random_state = 42

gmm = GMMHMM(
    n_components=n_components,
    n_mix=n_mix,
    covariance_type=cov_type,
    min_covar=min_covar,
    startprob_prior=startprob_prior,
    transmat_prior=transmat_prior,
    weights_prior=weights_prior,
    means_prior=means_prior,
    means_weight=means_weight,
    covars_prior=covars_prior,
    covars_weight=covars_weight,
    algorithm=algorithm,
    n_iter=n_iter,
    tol=tol,
    params=params,
    init_params=init_params,
    implementation=implementation,
    random_state=random_state,
)
gmm.fit(X)
latent_states = gmm.predict(X)

# 绘制拟合结果
sns.set_style("white")
plt.figure(figsize=(20, 8))
for i in range(gmm.n_components):
    state = latent_states == i
    plt.plot(datelist[state], closeidx[state], ".", label="latent state %d" % i, lw=1)
    plt.legend()
    plt.grid(1)
