[toc]
# 频率派VS贝叶斯派
## 问题定义
### 数据格式
$$
X = (\mathbf{x}_1, \mathbf{x}_2,...,\mathbf{x}_N)_{N \times P}^T = \left (
\begin{array}{cccc}
x_{11} & x_{12} & ... & x_{1p}\\
x_{21} & x_{22} & ... & x_{2p}\\
... & ... & ... & ...\\
x_{N1} & x_{N2} & ... & x_{Np}
\end{array}\right )_{N \times p}
$$
### 参数
模型所有的参数使用$\theta$表示
## 频率派
### 主要观点
模型的参数为未知的常量, 一个确定的最优值。
### 参数求解
主要通过极大似然估计(MLE)对参数值求解
$$\theta_{MLE} = \arg \max_{\theta} \prod_{i=1}^Np(\mathbf{x}_i|\theta)$$
## 贝叶斯派
### 主要观点
模型的参数服从一个分布，假定未观测数据之前对参数有一个先验分布$\theta \sim P(\theta)$, 观测数据$X$之后获得后验分布$P(\theta|X)$, 先验分布与后验分布通过似然函数联系起来,利用贝叶斯定理
$$
P(\theta|X) = \frac{P(X|\theta) P(\theta)}{P(X)} \propto P(X|\theta) P(\theta)
$$
### 参数求解
MAP:
$$\theta_{MAP} = \arg \max_{\theta} P(\theta |X) = \arg \max_{\theta} P(X|\theta)P(\theta)$$
对于新的数据$\hat{x}$的预测
$$
\begin{aligned}
p\left(\hat{x} | X\right) &= \int_{\theta} p\left(\hat{x}, \theta | X \right) d \theta \\
&= \int_{\theta} p\left(\hat{x} | \theta\right) \cdot p(\theta | X) d \theta \\
\end{aligned}
$$
# 对比

||人工智能应用领域|最终求解的问题|
|--|--|--|
|频率派|统计机器学习|优化问题|
|贝叶斯派|概率图模型|积分问题(常用求解方法: MCMC)|