[toc]
# 马尔科夫链蒙特卡洛方法 (Markov Chain & Monte Carlo, MCMC)
## 理论基础
### Monte Carlo Method 
基于采样的随机近似方法
#### 采样的动机
1. 采样本身就是常见的任务
2. 求和或求积分
#### 什么是好的样本
1. 样本趋近于高概率区域
2. 样本之间互相独立
#### ~采样困难性~
1. partation function is intractable
   无法有效的求出分布的累计分布函数 (Cumulative Distribution Function, CDF)
2. high dimension 
   样本维度过高导致采样空间极大, 导致采样非常困难。
   
#### 几种常见的采样方法
##### 概率分布采样
根据目标分布的累计分布函数 (Cumulative Distribution Function, CDF), 通过生成随机数(均匀分布)的方式进行采样。
##### Rejection Sampling
**对于复杂分布，我们仅仅能够获得概率密度函数（Probability density function, PDF）,无法获取CDF。**

我们首先选取一个简单分布 $q(z)$ (Proposed distribution), 使其满足
$$
\forall z_{i}, M q(z_i) \geq p(z_i)
$$

具体采样过程如下:
i: $z^{\left( i \right)} \sim q(z)$
ii: $u \sim U(0,1)$
    if $u \leq \alpha$, 接受 $z^{\left( i \right)}$
    else 拒绝 

其中 $\alpha$ 为接受率， $\alpha = \frac{p \left( z^{\left( i \right)} \right)}{M q \left( z^{\left( i \right)} \right)}, 0\leq \alpha \leq 1$

**当 $q$ 与 $p$ 差距过大的时候，采样效率会极低。**

##### Importance Sampling
为了提升 Rejection Sampling 的采样效率, Importance Sampling 通过如下方式采样:
$$
\begin{aligned}
E_{p \left( z \right)} [f \left( z \right)] &= \int p \left( z \right) f \left( z \right) d z \\
&= \int \frac{f \left( z \right)  p \left( z \right)}{q \left( z \right)} q \left( z \right)  d z\\
&= \frac{1}{N} \sum\limits_{i=1}^N f \left( z_i \right) \frac{p \left( z_i \right)}{q \left( z_i \right)}
\end{aligned}
$$
其中 $z_i \sim q \left( z \right)$, $\frac{p \left( z_i \right)}{q \left( z_i \right)}$ 对应第 $i$ 个样本的权重。

通过这种方式在采集同样数量的样本点，Importance Sampling 的效率远高于 Rejection Sampling。

**但是如果采样数过少的话，可能采集不到重要程度较高的样本。**
###### Sampling-Importance-Resampling

- 核心思想 
解决SIS(Sequential Importance Sampling)问题中权重退化的问题。
当权重分布极度不平衡的时候，直接采用Importance Sampling 效果不理想。
- 流程
在进行完 Importance Sampling 的基础上，利用每个样本及其权重重新估算 $q(z)$, 最终利用 $q(z)$ 重新采样。

##### 总结
Rejection Sampling 与 Importance Sampling 要求 q(x) 与 p(x) 接近， 并且 q(x) 较为简单并且易于采样。
### Markov Chain
时间与状态都是离散的随机过程
#### 齐次(一阶) Markov Chain
$$
\begin{aligned}
P \left( X_{t+1} = x| x_1, x_2,...,x_t \right) = P \left( X_{t+1} =x | X_t \right)\\
\end{aligned}
$$
$$
Q_{ij}: Q_{ij} = p \left( x_{t+1} = j | x_t = i \right)
$$
#### 平稳分布
$\left\{ \pi \left( k \right) \right\}$ 是 $\left\{ x_t \right\}$ 的平稳分布，需要满足一下条件:
$$
\begin{aligned}
&\pi \left( x^{*} \right) = \int \pi \left( x \right) Q \left( x^{*} | x \right) dx\\
&s.t. \quad \left  \{
\begin{array}{l}
\pi = \left[ \pi(1), \pi(2),...,\pi(N),... \right] \\
\sum\limits_{i=1}^{\infty} \pi (i) = 1
\end{array}
\right. 
\end{aligned}
$$
##### 平稳分布证明
Q 矩阵为随机矩阵， 因此Q矩阵的特征值为 $\leq$ 1, $Q = A \Lambda A^{-1}$
其中 $\Lambda = \left (\begin{array}{cccc}\lambda_1 & & & \\& \lambda_2 & & \\& & ... & \\& & & \lambda_K\end{array}\right )$, $\lambda_i \leq 1, \text{for i} = 1,2,3,...,K$ 
则 $\pi_{t+1} = \pi_1 (A \Lambda A^{-1})^t=q_1 (A \Lambda^t A^{-1})$, 存在足够大的 m， 使得 $\Lambda^{m} = \left (\begin{array}{cccc}0 & & & \\& 1 & & \\& & ... & \\& & & 0\end{array}\right )$ 。
$$
\begin{aligned}
\pi_{m+1} &= \pi_1 A \Lambda^m A^{-1}\\
\pi_{m+2} &= \pi_{m+1} (A \Lambda A^{-1})\\
&= \pi_1 A \Lambda^m \Lambda A^{-1}\\
&= \pi_1 A \Lambda^m A^{-1}\\
&= \pi_{m+1}
\end{aligned}
$$
因此当 $t > m$, $\pi_{m+1} = \pi_{m+2} = \cdots = \pi_t = \cdots$.

#### Detailed Balance:
$$
\pi (x) P \left( x\longrightarrow x^{*} \right) = \pi \left( x^{*} \right) P \left( x^{*} \longrightarrow x \right)
$$
该条件为平稳分布的充分非必要条件。
##### 证明
$$
\begin{aligned}
&\int \pi \left( x \right) P \left( x \longrightarrow x^{*} \right) dx\\
&= \int \pi \left( x^{*} \right) P \left( x^{*} \longrightarrow x \right)dx\\
&= \pi \left( x^{*} \right) \int P \left( x^{*} \longrightarrow x \right)dx\\
&= \pi \left( x^{* } \right)
\end{aligned}
$$
## Metropolis-Hastings
### 核心思想
**构建一个齐次马尔科夫链逼近q(z)。**
### 推导
为了使采样过程满足平稳分布, 我们令:
$$
P \left( z \right) Q \left( z\rightarrow z^{*} \right) \alpha \left( z, z^{*} \right) = P \left( z^{*} \right) Q \left( z^{*} \rightarrow z \right) \alpha \left( z^{*} ,z \right)
$$
其中 $\alpha =  \min \left( 1, \frac{P \left( z^{*} \right)Q \left( z^{*} \rightarrow z \right) }{P \left( z \right) Q \left( z\rightarrow z^{*} \right)} \right)$ 。
#### $\alpha$ 证明
$$
\begin{aligned}
&P \left( z \right) Q \left( z \rightarrow z^{*} \right) \alpha \left( z, z^{*} \right)\\
&= P \left( z \right) Q \left( z \rightarrow z^{*} \right) \min \left( 1, \frac{P \left( z^{*} \right)Q \left( z^{*} \rightarrow z \right) }{P \left( z \right) Q \left( z\rightarrow z^{*} \right)} \right)\\
&=\min \left( P(z) Q(z\rightarrow z^{*}) \right) P \left( z^{*} \right) Q (z^{*}\rightarrow z)) \\
&= P \left( z^{*} \right) Q \left( z^{*} \rightarrow z \right) \min \left( 1, \frac{P \left( z \right)Q \left( z \rightarrow z^{*} \right) }{P \left( z^{*} \right) Q \left( z^{*} \rightarrow z \right)} \right)\\
&= P \left( z^{*} \right) Q \left( z^{*} \rightarrow z \right) \alpha \left( z^{*} ,z \right)
\end{aligned}
$$
### 采样过程
$$
\begin{aligned}
& \text{for t} : 1\longrightarrow M\\
&\quad \text{for i} : 1\longrightarrow N\\
&\qquad u \sim U \left( 0,1 \right)\\
&\qquad z^{*} \sim Q \left( z_{t-1}^{(i)}\rightarrow z^{*} \right)\\
&\qquad \alpha =  \min \left( 1, \frac{P \left( z^{*} \right)Q \left( z^{*} \rightarrow z_{t-1}^{(i)} \right) }{P \left( z_{t-1}^{(i)} \right) Q \left( z_{t-1}^{(i)}\rightarrow z^{*} \right)} \right)\\
&\qquad \text{if } u \leq  \alpha, z_{t}^{\left( i \right)} = z^{*}\\
&\qquad \text{else } z_{t}^{\left( i \right)} = z_{t-1}^{\left( i \right)} 
\end{aligned}
$$
其中 $z_{t}^{(i)}$ 表示第 $t$ 次迭代中的第 $i$ 个样本对应的值, $z_t = \left\{ z_t^{(1)},z_t^{(2)},...,z_t^{(N)}\right\}$ , $M$ 表示迭代次数, $N$ 表示样本数量。
## Gibbs Sampling
### 核心思想
通过设计转移矩阵令接受率为 1 的 Metroplits-Hastings.
### 推导
令 $Q(x \rightarrow x^{*}) = P \left( z^{(i*)} | z^{(-i)} \right)$, 可以进一步推导接受率 $\alpha$
$$
\begin{aligned}
\alpha &=  \min \left( 1, \frac{P \left( z^{*} \right)Q \left( z^{*} \rightarrow z \right) }{P \left( z \right) Q \left( z\rightarrow z^{*} \right)} \right)\\
&= \min \left( 1, \frac{P(z^{i*}|z^{-i})P \left( z^{-i} \right)Q \left( z^{i*} \rightarrow z^{i}  \right)}{P(z^{i}|z^{-i})P \left( z^{-i} \right)Q \left( z^{i} \rightarrow z^{i*}  \right)} \right)\\
&= \min \left( 1, \frac{P(z^{i*}|z^{-i})P \left( z^{-i} \right) P \left( z^{i} | z^{-i}   \right)}{P(z^{i}|z^{-i})P \left( z^{-i} \right)Q \left( z^{i*} | z^{-i}  \right)} \right)\\
&= \min (1,1)\\
&= 1
\end{aligned}
$$

## MCMC存在的问题
1. 理论只保证收敛性，但无法知道何时收敛
2. mixing time 过长
   1. q(z) 本身过于复杂
   2. 维度过高，各个维度之间存在相关性 $z_t^{\left( i \right)} = [z_{1,t}^{\left( i \right)}, z_{2,t}^{\left( i \right)},..., z_{K,t}^{\left( i \right)}]$, K 过大
3. 样本之间有一定的相关性 $z_t$ 与 $z_{t+1}$ 之间存在相关性。
 
### 名词解释
burn-in 达到平稳分布之前的转移过程
mixing time 达到平稳分布之前的转移时间
   

