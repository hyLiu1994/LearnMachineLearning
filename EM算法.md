[toc]
# EM算法
主要用于解决具有隐变量的混合模型的参数估计(最大似然)。
- 对于比较简单的问题，可以直接获取解析解。
$$\hat{\theta} = \arg \max_{\theta} p(x| \theta) \propto  \arg \max_{\theta} \log p(x| \theta)$$
- 对于无法求出解析解，或者解析解比较难求的问题(GMM)，就会采用迭代算法。例如对于观测变量为Y, 隐变量为Z(Z为离散型的分布)，我们使用极大似然估计的求解步骤如下
$$\begin{aligned}
\hat{\theta} &= \arg \max_{\theta} p(x| \theta) \\
 &= \arg \max_{\theta} \log p(x| \theta) \\
  &= \arg \max_{\theta} \log \sum_z p(x, z| \theta) \\
 &=  \arg \max_{\theta} \log\left[\sum_zp(x|z, \theta)p(z|\theta)\right]\\
\end{aligned}$$
可以看出 $\log$ 包含了求和，这为求出解析解带来了困难。EM算法使用迭代的方式求取近似解，规避了解析解难以求出的问题。
## EM 迭代公式
定义 $\theta^{(t+1)}$ 如下：
$$
\theta^{(t+1)} = \arg \max_{\theta} \int_z \log p(x, z|\theta) \cdot p(z|x, \theta^{(t)})dz
$$
其中 $\int_z \log p(x, z|\theta) \cdot p(z|x, \theta^{(t)})dz$, 也可以看作$E_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]$

## EM 迭代公式证明
试证明:
若 $\theta^{(t)} \rightarrow \theta^{(t+1)}$ :
则
$$
\log p(x|\theta^{(t)}) \leq \log p(x| \theta^{(t+1)})
$$
### 证明1
证明如下:
$$\log p(x|\theta) = \log p(x, z|\theta) - \log p(z|x, \theta)$$
> $$
\begin{aligned}
& p(x | \theta) p(z|x, \theta) = p(z, x| \theta) \\
& \Rightarrow \log p(x| \theta) + \log p(z|x, \theta) = \log p(x, z| \theta) \\ 
& \Rightarrow \log p(x|\theta) = \log p(x, z|\theta) - \log p(z|x, \theta)
\end{aligned}
$$

这里左右同时求一个关于 $p(z|x, \theta^{(t)})$ 的积分
- 左侧
$$
\begin{aligned}
&\text{左侧} = \int p(z|x, \theta^{(t)}) \log p(x|\theta)dz = \log p(x|\theta) \int p(z|x, \theta^{(t)})dz\\
&\text{由于} p(z|x, \theta^{(t)}) \text{是仅关于} z 的函数，所以 \int p(z|x, \theta^{(t)})dz = 1 \\
&\text{即}\\
&\text{左侧} = \log p(x|\theta)
\end{aligned}
$$
- 右侧
$$\begin{aligned}
&\text{右侧}=\int p(z|x, \theta^{(t)}) \log p(x, z|\theta)dz - \int p(z|x, \theta^{(t)}) \log p(z|x, \theta)dz
\end{aligned}
$$
记
$$
\begin{aligned}
Q(\theta, \theta^{(t)}) = \int p(z|x, \theta^{(t)}) \log p(x, z|\theta)dz \\
H(\theta, \theta^{(t)}) = \int p(z|x, \theta^{(t)}) \log p(z| x, \theta)dz
\end{aligned}
$$
则
$$
\begin{aligned}
& \log p(x|\theta) = Q(\theta, \theta^{(t)}) - H(\theta, \theta^{(t)})\\
&\log p(x|\theta^{(t+1)}) = Q(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t+1)}, \theta^{(t)})\\
&\log p(x|\theta^{(t)}) = Q(\theta^{(t)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})
\end{aligned}
$$
所以欲证明原命题成立，即是证明

$$
\begin{aligned}
&Q(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t+1)}, \theta^{(t)})  \geq Q(\theta^{(t)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)}) \\
&\Leftrightarrow Q(\theta^{(t+1)}, \theta^{(t)}) - Q(\theta^{(t)}, \theta^{(t)}) \geq H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})
\end{aligned}
$$
由 $\theta^{(t+1)}$ 定义可知
$$
Q(\theta^{(t+1)}, \theta^{(t)}) \geq  Q(\theta, \theta^{(t)})
$$
所以 
$$
Q(\theta^{(t+1)}, \theta^{(t)}) \geq  Q(\theta^{(t)}, \theta^{(t)})
\Rightarrow Q(\theta^{(t+1)}, \theta^{(t)}) -  Q(\theta^{(t)}, \theta^{(t)}) \geq 0
$$
由 
$$
\begin{aligned}
&H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})\\
&= \int p(z|x, \theta^{(t)})  \left[ \log p(z| x, \theta^{(t+1)}) - \log p(z|x,\theta^{(t)}) \right]dz \\
&= \int p(z|x, \theta^{(t)})  \left[ \log \frac{p(z| x, \theta^{(t+1)})}{p(z| x, \theta^{(t)})} \right]dz = - KL\left[p(z|x, \theta^{(t)})||p(z| x, \theta^{(t+1)}) \right]\\
\end{aligned}
$$
由 Jassen 不等式, 对$f(x) = \log(x)$, 有
$$
E[\log(x)] \leq \log(E[x])
$$
所以
$$
\begin{aligned}
&H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})\\
&= \int p(z|x, \theta^{(t)})  \left[ \log \frac{p(z| x, \theta^{(t+1)})}{p(z| x, \theta^{(t)})} \right]dz \\
&\leq \int p(z|x, \theta^{(t)}) \frac{p(z| x, \theta^{(t+1)})}{p(z| x, \theta^{(t)})}dz \\
&= \log \left[\int p(z| x, \theta^{(t+1)})dz\right] \\
&=0
\end{aligned}
$$
所以
$$
Q(\theta^{(t+1)}, \theta^{(t)}) - Q(\theta^{(t)}, \theta^{(t)}) \geq 0 \geq H(\theta^{(t+1)}, \theta^{(t)}) - H(\theta^{(t)}, \theta^{(t)})
$$
所以原问题 得证。
### 证明2
(此处证明与证明1类似, 步骤更紧凑一些)
$$
\begin{aligned}
& p(x | \theta) p(z|x, \theta) = p(z, x| \theta) \\
& \Rightarrow \log p(x| \theta) + \log p(z|x, \theta) = \log p(x, z| \theta) \\ 
& \Rightarrow \log p(x|\theta) = \log p(x, z|\theta) - \log p(z|x, \theta) \\
& \text{左右同时求一个关于} p(z|x, \theta^{(t)}) \text{的积分}\\
& \Rightarrow \int \log p(x|\theta) p(z|x, \theta^{(t)})d z = 
\int \left[\log p(x, z|\theta) - \log p(z|x, \theta)\right] p(z|x, \theta^{(t)})d z \\
& \Rightarrow \int \log p(x|\theta) p(z|x, \theta^{(t)})d z = 
\int \log p(x, z|\theta) p(z|x, \theta^{(t)})d z - \int \log p(z|x, \theta)p(z|x, \theta^{(t)})d z \\
& \Rightarrow \log p(x|\theta) = \int \log p(x, z|\theta) p(z|x, \theta^{(t)})d z - \int \log p(z|x, \theta)p(z|x, \theta^{(t)})d z \\
& \text{所以} \\
& \log p(x|\theta^{(t+1)}) - \log p(x|\theta^{(t)}) = \left[\int \log p(x, z|\theta^{(t+1)}) p(z|x, \theta^{(t)})d z - \int \log p(x, z|\theta^{(t)}) p(z|x, \theta^{(t)})d z\right]\\ &  - \int \log \frac{p(z|x, \theta^{(t+1)})}{p(z|x, \theta^{(t)})} p(z|x, \theta^{(t)})d z \\
& \text{由} \theta^{(t+1)} \text{定义可得} \\
&\left[\int \log p(x, z|\theta^{(t+1)}) p(z|x, \theta^{(t)})d z - \int \log p(x, z|\theta^{(t)}) p(z|x, \theta^{(t)})d z\right] \geq 0 \\
& \text{由 Jasson 不等式}\\
& \int \log \frac{p(z|x, \theta^{(t+1)})}{p(z|x, \theta^{(t)})} p(z|x, \theta^{(t)})d z \leq \log \left[\int \frac{p(z|x, \theta^{(t+1)})}{p(z|x, \theta^{(t)})} p(z|x, \theta^{(t)})d z\right] = 0 \\
& \text{综上} \\
& \log p(x|\theta^{(t+1)}) - \log p(x|\theta^{(t)}) \geq 0 \\
& \text{问题得证}
\end{aligned}
$$

## EM 求解问题步骤
### 符号定义
$x$: 观测数据 (observed data)
$z$: 因变量 (latent variable; unobserved data)
$(x, z)$: complete data
$\theta$: 参数
### 问题求解
#### 迭代公式
$$
\theta^{(t+1)} = \arg \max_{\theta} \int_z \log p(x, z|\theta) \cdot p(z|x, \theta^{(t)})dz
$$
#### E-step：
$$
p(z|x, \theta^{(t)}) \rightarrow E_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]
$$
#### M-step:
$$
\theta^{(t+1)} = \arg \max_{\theta}  E_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]
$$
### 公式由来 - KL散度角度
$$
\begin{aligned}
\log p(x|\theta) &= \log p(x, z|\theta) - \log p(z|x, \theta) \\
&=\log \frac{p(x, z|\theta)}{q(z)} - \log \frac{p(z|x, \theta)}{q(z)}\\
\end{aligned}$$
$$
\begin{aligned}
&\text{左边} = \int_z \log p(x|\theta) q(z)d z \\
&= \log p(x|\theta) \int_z  q(z)d z  \\
&= \log p(x|\theta) \\
&\text{右边} = \underbrace{ \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z}_{ELBO(evidence\ lower\ bound)} \underbrace{- \int_{z} q(z) \log \frac{p(z|x, \theta)}{q(z)} d z}_{KL \left[q(z)||p(z|x, \theta\right)]}\\
&\text{综上}\\
&\log p(x|\theta) = ELBO + KL \left[q(z)||p(z|x, \theta\right)]\\
&\because KL \left[q(z)||p(z|x, \theta\right)] \geq 0\\
&\therefore \log p(x|\theta) \geq ELBO, \text{等号仅在} q(z) \text{与} p(z|x, \theta) 相同时，成立\\
\end{aligned}$$
由以上分析可知, 可以通过不断最大化 $ELBO$ 的方式, 最大化 $\log p(x|\theta)$ 
$$
\begin{aligned}
\hat{\theta} &= \arg \max_{\theta} ELBO \\
&=\arg \max_{\theta}  \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z\\
&\text{令} q(z) = p(z|x, \theta^{(t)}) \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \log \frac{p(x, z | \theta)}{p(z|x, \theta^{(t)})} d z \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \left[\log p(x, z | \theta)- \log p(z|x, \theta^{(t)})\right] d z \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \log p(x, z | \theta) d z \\
\end{aligned}
$$
### 公式由来 - Jassen 不等式角度
$$\begin{aligned}
\log P(x | \theta) &=\log \int_{z} P(x, z | \theta) d z \\
&=\log \int_{z} \frac{P(x, z | \theta)}{q(z)} q(z) d z \\
&=\log E_{q(z)}\left[\frac{p(x, z | \theta)}{q(z)}\right] \\
& \geq E_{q(z)}\left[\log \frac{p(x, z | \theta)}{q(z)}\right] \\
&\text{当且仅当} \frac{p(x, z | \theta)}{q(z)} = c, \text{时, 取等号}\\
& q(z) = \frac{1}{c} p(x, z | \theta) \\
&\Rightarrow \int q(z) d z = \frac{1}{c} \int  p(x, z | \theta) d z \\
&\Rightarrow 1 =  \frac{1}{c} p(x| \theta)\\
&\Rightarrow c = p(x| \theta) \\
&\Rightarrow q(z) = \frac{p(x, z | \theta)}{p(x| \theta)}  = p(z|x, \theta)
\end{aligned}$$
由以上分析可知, 可以通过不断最大化 $E_{q(z)}\left[\log \frac{p(x, z | \theta)}{q(z)}\right]$ 的方式, 最大化 $\log p(x|\theta)$ 
$$
\begin{aligned}
\hat{\theta} &= \arg \max_{\theta} E_{q(z)}\left[\log \frac{p(x, z | \theta)}{q(z)}\right]\\
&=\arg \max_{\theta}  \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z\\
&\text{令} q(z) = p(z|x, \theta^{(t)}) \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \log \frac{p(x, z | \theta)}{p(z|x, \theta^{(t)})} d z \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \left[\log p(x, z | \theta)- \log p(z|x, \theta^{(t)})\right] d z \\
&=\arg \max_{\theta}  \int_{z} p(z|x, \theta^{(t)}) \log p(x, z | \theta) d z \\
\end{aligned}
$$
# 广义 EM
由前节推导可知， EM 算法步骤如下
- 迭代公式
$$
\theta^{(t+1)} = \arg \max_{\theta} \int_z \log p(x, z|\theta) \cdot p(z|x, \theta^{(t)})dz
$$
- E-step：
$$
p(z|x, \theta^{(t)}) \rightarrow E_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]
$$
- M-step:
$$
\theta^{(t+1)} = \arg \max_{\theta}  E_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]
$$
在 **E-step**, 主要目的是求出后验分布 $p(z|x, \theta^{(t)})$, 进而求出 $\log p(x, z|\theta)$ 关于 $p(z|x, \theta^{(t)})$ 的期望。
在 **M-step**, 主要目的是求出， **E-step**步求得的期望下，求取最优 $\theta$
由前节推导可知
$$
\begin{aligned}
&\log p(x|\theta) = \underbrace{ \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z}_{ELBO(evidence\ lower\ bound)} \underbrace{- \int_{z} q(z) \log \frac{p(z|x, \theta)}{q(z)} d z}_{KL \left[q(z)||p(z|x, \theta\right)]}\\
&\log p(x|\theta) = ELBO + KL \left[q(z)||p(z|x, \theta\right)]\\
\end{aligned}
$$

在之前的推导中，我们知道 $q$ 的最优分布为 $p(z|x, \theta)$， 但是在现实的问题中，由于问题过于复杂，有可能 $p(z|x, \theta)$ 的分布无法求出。在此种情况下，我们作如下讨论：
由于 对于 $ELBO$ 来讲，未知参数为 $q, \theta$, 不妨假设 $EBLO = L(q, \theta)$
- 若我们固定 $\theta$ , 则 $\log p(x|\theta)$ 为一个已知常数， 我们需要 $\log p(x|\theta)$ 的最优下界 $ELBO$,那么,最优的 $\hat{q}$ 为
$$\hat{q} = \arg \min_{q} KL \left[q(z)||p(z|x, \theta\right)] \Leftrightarrow \hat{q} = \arg \max_{q} L(q, \theta) $$
> 当待解决问题较为简单时, 即 $\hat{q} = \arg \min_{q} KL \left[q(z)||p(z|x, \theta\right)]$ 可以求出解析解时，易知，此时最优解为 $q(z) = p(z|x, \theta)$
- 同样的，若我们固定 $q$ , 则 最优的 $\hat{\theta}$ 为
$$ \hat{\theta} = \arg \max_{q} L(\hat{q}, \theta) $$
## 总结
广义的EM结构如下
- 问题定义
$$
\begin{aligned}
&\log p(x|\theta) = \underbrace{ \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z}_{ELBO(evidence\ lower\ bound)} \underbrace{- \int_{z} q(z) \log \frac{p(z|x, \theta)}{q(z)} d z}_{KL \left[q(z)||p(z|x, \theta\right)]}\\
&\log p(x|\theta) = \underbrace{ ELBO }_{L(q, \theta)} + KL \left[q(z)||p(z|x, \theta\right)]\\
\end{aligned}
$$
因 $KL \left[q(z)||p(z|x, \theta\right)] \geq 0 $ 
则, $L(q, \theta)$ 为 $\log p(x|\theta)$ 的下界， 所以可以通过最大化 $L(q, \theta)$ 的方式，最大化 $\log p(x|\theta)$, 即
$$
\arg \max_{\theta} \quad \log p(x|\theta) \Leftarrow \arg \max_{\theta} \quad L(q, \theta)
$$
- 算法步骤
$$
\left\{\begin{array}{c} \text{E-step:} \quad q^{(t+1)} = \arg \max_{q} L(q, \theta^{(t)})\\ 
\text{M-step:} \quad \theta^{(t+1)} = \arg \max_{q} L(q^{(t+1)}, \theta)\\ 
\end{array}\right.
$$
由前节分析可知
$$
\begin{aligned}
L(q, \theta) &= \int_{z} q(z) \log \frac{p(x, z | \theta)}{q(z)} d z \\
&=\int_{z} q(z) \log p(x, z | \theta) d z - \int_{z} q(z) \log p(z) d z \\
&= E_{q(z)}\left[\log p(x, z | \theta)\right] + \int_{z} q(z) \log \frac{1}{p(z)} d z \\
&= E_{q(z)}\left[\log p(x, z | \theta)\right] + H[q(z)]
\end{aligned}$$

若 $q(z)$ 是可以求出的，那么，$H\left[q(z)\right]$ 为已知，该部分不影响后续优化， 例如在第一节的优化中，假定该分布已知，在后续优化部分中不包含该部分。
# EM 算法变种
## EM 算法
- 问题定义
$$
\hat{\theta} = \arg \max_{\theta}p(x|\theta) = \arg \max_{\theta} log p(x|\theta)
$$
- 问题求解
$$
\log p(x|\theta) = \underbrace{ELBO}_{L(q, \theta)} + KL \left[q(z)||p(z|x, \theta\right)] \geq L(q, \theta)
$$
$$
\left\{\begin{array}{c} \text{E-step:} \quad q^{(t+1)} = \arg \max_{q} L(q, \theta^{(t)})\\ 
\text{M-step:} \quad \theta^{(t+1)} = \arg \max_{q} L(q^{(t+1)}, \theta)\\ 
\end{array}\right.
$$
EM 也可以从 坐标上升法(例如 SMO) 角度来看,即，首先固定一部分维度，优化另一部分，然后在两类维度之间不断交替优化。
- 变种： E-step 中 当 $q$ 难以获取时， 使用 VI 或 VB(变分推断) 可以获取 $q$ 的近似分布，这样变种一般称为 **VBEM**, **VEM**；当使用采样方法后去 $q$ 的分布的时候(例如 蒙特卡罗方法)，我们称其为 **MCEM** 
## 遗留问题
KL散度性质
