# 实验设计与方法

本文档详细介绍了独立柯西分布KL散度不等式验证实验的设计思路、方法和实现细节。

## 1. 实验目标

通过系统性的数值实验，验证以下不等式在不同参数配置下的成立情况：

$$ D_{KL}(P_0 \| P_{fused}) \le N \sum_{k=1}^N D_{KL}(P_0 \| P_k) $$

其中：
- $P_0$ 是基准 $d$-维独立柯西分布
- $P_k$ 是扰动分布
- $P_{fused}$ 是融合分布，其参数通过简单线性叠加各个扰动量的总和得到
- $N$ 是扰动分布的数量

## 2. 实验设计思路

为全面验证不等式的成立情况，我们设计了以下六组实验：

1. **仅位置参数扰动实验**：固定尺度参数，仅改变位置参数
2. **仅尺度参数扰动实验**：固定位置参数，仅改变尺度参数
3. **双参数扰动实验**：同时改变位置和尺度参数
4. **多维情况实验**：考虑不同维度 $d$ 下的情况
5. **极端情况测试**：测试参数取极端值时的情况
6. **最优常数探索**：寻找更一般形式不等式中的最优常数 $C$

每组实验都考虑了不同的扰动分布数量 $N$ 和不同的参数配置，以全面覆盖参数空间。

## 3. 实验方法

### 3.1 KL散度计算

对于一维柯西分布，KL散度的计算公式为：

$$ D_{KL}(P_0 \| P) = \log\left( \frac{(\gamma_0 + \gamma)^2 + (\mu_0 - \mu)^2}{4 \gamma_0 \gamma} \right) $$

对于多维独立柯西分布，KL散度为各维度KL散度之和：

$$ D_{KL}(P_0 \| P) = \sum_{i=1}^d D_{KL}(P_{0i} \| P_i) $$

### 3.2 参数设置

- **位置参数扰动范围**：$\Delta\mu \in [0.1, 0.5, 1, 2, 5, 10, 50, 100]$
- **尺度参数扰动范围**：$\Delta s \in [-2, -1, -0.5, 0.5, 1, 2]$
- **扰动分布数量**：$N \in [2, 3, 5, 10]$
- **维度**：$d \in [1, 2, 5, 10]$
- **极端值**：$[10^2, 10^3, 10^4, 10^5]$

### 3.3 扰动模式

对于每组参数配置，我们考虑了两种扰动模式：

1. **同向扰动**：所有扰动分布的参数调整方向相同
2. **反向扰动**：扰动分布的参数调整方向相反

### 3.4 实验指标

- **不等式成立比例**：在所有测试情况中，不等式成立的比例
- **KL散度比率**：$\frac{D_{KL}(P_0 \| P_{fused})}{N \sum_{k=1}^N D_{KL}(P_0 \| P_k)}$，该比率小于1表示不等式成立

## 4. 实验实现

实验使用Python实现，主要依赖numpy、matplotlib、pandas等库。核心代码结构如下：

```python
def kl_divergence_cauchy(mu_0, gamma_0, mu, gamma):
    """计算一维柯西分布之间的KL散度"""
    return np.log((gamma_0 + gamma)**2 + (mu_0 - mu)**2) - np.log(4 * gamma_0 * gamma)

def kl_divergence_multivariate_cauchy(mu_0, gamma_0, mu, gamma):
    """计算多维独立柯西分布之间的KL散度"""
    d = len(mu_0)
    kl_sum = 0
    for i in range(d):
        kl_sum += kl_divergence_cauchy(mu_0[i], gamma_0[i], mu[i], gamma[i])
    return kl_sum

def experiment_1d_position_only(N_values, delta_mu_values, gamma_0=1.0):
    """实验1：一维情况下，仅位置参数扰动的情况"""
    results = []
    
    for N, delta_mu in product(N_values, delta_mu_values):
        # 同向扰动
        delta_mu_k = np.ones(N) * delta_mu
        mu_fused = np.sum(delta_mu_k)
        
        # 计算融合分布的KL散度
        kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_0)
        
        # 计算每个扰动分布的KL散度
        kl_sum = 0
        for k in range(N):
            kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_0)
        
        # 计算不等式右侧
        right_side = N * kl_sum
        
        # 记录结果
        results.append({
            'N': N,
            'delta_mu': delta_mu,
            'pattern': 'same_direction',
            'kl_fused': kl_fused,
            'right_side': right_side,
            'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
            'inequality_holds': kl_fused <= right_side
        })
        
        # 反向扰动
        if N > 1:
            delta_mu_k = np.zeros(N)
            delta_mu_k[0] = delta_mu
            delta_mu_k[1:] = -delta_mu / (N-1)
            mu_fused = np.sum(delta_mu_k)
            
            # 计算融合分布的KL散度
            kl_fused = kl_divergence_cauchy(0, gamma_0, mu_fused, gamma_0)
            
            # 计算每个扰动分布的KL散度
            kl_sum = 0
            for k in range(N):
                kl_sum += kl_divergence_cauchy(0, gamma_0, delta_mu_k[k], gamma_0)
            
            # 计算不等式右侧
            right_side = N * kl_sum
            
            # 记录结果
            results.append({
                'N': N,
                'delta_mu': delta_mu,
                'pattern': 'opposite_direction',
                'kl_fused': kl_fused,
                'right_side': right_side,
                'ratio': kl_fused / right_side if right_side > 0 else float('inf'),
                'inequality_holds': kl_fused <= right_side
            })
    
    return pd.DataFrame(results)
```

完整实验代码包含六个主要实验函数和一个运行所有实验的主函数，详见项目代码库。

## 5. 实验流程

1. 设置实验参数（N值、维度d、参数扰动范围等）
2. 对每组参数配置，计算融合分布和各扰动分布的KL散度
3. 计算不等式左右两侧的值，并判断不等式是否成立
4. 记录实验结果，包括KL散度值、比率和不等式成立情况
5. 对结果进行统计分析和可视化

通过这一系统性的实验设计和实现，我们能够全面验证KL散度不等式在不同参数配置下的成立情况，为理论分析提供实证支持。
