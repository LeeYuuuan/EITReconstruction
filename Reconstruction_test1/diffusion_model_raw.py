import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
import torch
import torch.nn as nn

# 生成一万个点，得到s curve
s_curve, _ = make_swiss_roll(10**4,noise=0.1)
s_curve = s_curve[:,[0,2]]/10.0
print("shape of s:",np.shape(s_curve))
data = s_curve.T
# 构建数据集
dataset = torch.Tensor(s_curve).float()

#确定超参数的值
num_steps = 100 

#制定每一步的beta，beta按照时间从小到大变化
betas = torch.linspace(-6,6,num_steps)  # 使用linespace指定bata的值，也可以用其他方法
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

alphas = 1-betas

alphas_prod = torch.cumprod(alphas,0)


alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)

one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
    alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
        ==one_minus_alphas_bar_sqrt.shape

def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    #生成正态分布采样
    noise = torch.randn_like(x_0)
    #得到均值方差，根据时间步选择alphas_bar_sqrt值和one_minus_alphas_bar_sqrt值
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    #根据x0求xt
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声

num_shows = 20
fig,axs = plt.subplots(2,10,figsize=(28,3))
plt.rc('text',color='black')
for i in range(num_shows):
    j = i//10
    k = i % 10
    q_i = q_x(dataset,torch.tensor([i*num_steps//num_shows])) # 生成t时刻的采样数据，dataset是起始的x_0, q_x(x_0, t)
    axs[j,k].scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white')
    axs[j,k].set_axis_off()
    axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
plt.show()

