import numpy as np
from VMD import VMD
from sampleEntropy import sampleEntropy
from scipy.stats import pearsonr


def fitness1(Positions, DATA):
    """综合评价指标"""
    X = DATA['data']
    alpha = round(Positions[0])  # round(Positions(1, 1))
    K = round(Positions[1])  # round(Positions(1, 2))
    tau = DATA['tau']
    DC = DATA['dc']
    init = DATA['init']
    tol = DATA['tol']

    u, u_hat, omega = VMD(X, alpha, tau, K, DC, init, tol)

    ## 提取信号熵特征指标
    feature = np.zeros(K)  # feature数组初始化
    for ii in range(K):  # ii=1:K
        m = 2  # 维度一般取2
        r = 0.2 * np.std(u[ii, :])  # r=0.2*std(u(ii,:))
        tau = 1  # 是否降采样，1和2，选1
        feature[ii] = sampleEntropy(u[ii, :], m, r, tau)  # 样本熵

    Y = np.sum(u, axis=0)  # Y=sum(u,1)
    pear = pearsonr(X, Y)[0]  # pear=corr(X',Y','type','pearson') 皮尔逊系数
    d, _ = omega.shape  # [d,~]=size(omega) 综合惩罚
    D = np.log10(d)  # D=log10(d) 对数惩罚综合惩罚
    fitness1_value = np.min((feature / pear) * D)  # fitness1= min((feature/pear)*D)

    return fitness1_value