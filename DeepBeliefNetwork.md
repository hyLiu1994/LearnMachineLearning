# 深度信念网络 Deep Belief Network
## 核心思想
 相比于单层的受限玻尔兹曼机，其没有解决新的问题，但其具备更好的性能(更高的ELBO)。  
 
## 模型定义
![x](./Figure/DeepBeliefNetwork.png)
$$
\begin{aligned}
P \left( v, h^{(1)}, h^{(2)}, h^{(3)} \right) &= P \left( v | h^{(1)}, h^{(2)}, h^{(3)} \right) P \left( h^{(1)}, h^{(2)}, h^{(3)} \right)\\
&= P \left( v | h^{(1)} \right) P \left( h^{(1)}, h^{(2)}, h^{(3)} \right)\\
&= P \left( v | h^{(1)} \right) P \left( h^{(1)} |  h^{(2)} \right) P \left( h^{(2)}, h^{(3)} \right)\\
&= \prod\limits_{i} P \left( v_i | h^{(1)} \right) \prod\limits_{j} P \left( h_j^{(1)} | h^{(2)} \right) P \left( h^{(2)}, h^{(3)} \right)\\
\end{aligned}
$$
$$
\begin{aligned}
P \left( v_i | h^{(1)} \right) &= sigmoid \left( w_{:,i}^{T_{(1)}} h^{(1)} + b_i^{(0)} \right)\\
P \left( h_j^{(1)} | h^{(2)} \right) &= sigmoid \left( w_{:,j}^{T_{(2)}} h^{(2)} + b_j^{(1)} \right)\\
P \left( h^{(2)}, h^{(3)} \right) &= \frac{1}{Z} \exp \left\{ h^{T_{(3)}} w^{(3)} h^{(2)} + h^{T_{(2)}} b^{(2)} + h^{T_{(3)}} b^{(3)} \right\}
\end{aligned}
$$
## 模型有效性证明
$$
\begin{aligned}
\log P \left( v \right) &= \log \sum\limits_{h^{(1)}} P \left( v, h^{(1)} \right)\\
&= \log \sum\limits_{h^{(1)}} q \left( h^{(1)} | v \right) \frac{P \left( v, h^{(1)} \right)}{q \left( h^{(1)} | v \right)}\\
&= \log E_{q \left( h^{(1)} | v \right)} \left[ \frac{P \left( v, h^{(1)} \right)}{q \left( h^{(1)} | v \right)} \right]\\
&\geq E_{q(h^{(1)})} \log \frac{P \left( v, h^{(1)} \right)}{q \left( h^{(1)} | v \right)}\\
&= \sum\limits_{h^{(1)}} q \left( h^{(1)} | v \right) [\log P \left( v, h^{(1)} - \log q \left( h^{(1)} | v \right) \right)]\\
&= \sum\limits_{h^{(1)}} q \left( h^{(1)} | v \right) \underbrace{\log P \left( h^{(1)} \right)}_{\text{improve }P \left( h(1) \right)} + C 
\end{aligned}
$$
在 Belief Network 基础上添加玻尔兹曼机实际上是在优化 $P \left( h^{(1)} \right)$, 进而提高了ELBO.

## 后验分布的近似
$$
\begin{aligned}
&\log P \left( v \right) \geq ELBO = \sum\limits_{h^{(1)}} q \left( h^{(1)} | v \right) \log P \left( v, h^{(1)} \right) - \sum\limits_{h^{(1)}} q \left( h^{(1)} | v \right) \log q \left( h^{(1)} | v \right)\\
& q \left( h^{(1)} | v \right) = \prod\limits_{ i } q \left( h_i^{(1)} | v \right) = \prod\limits_{ i } sigmoid \left( w_{i,:}^{(1)} v + b_i^{(1)} \right)\\
\end{aligned}
$$
DBN $\rightarrow$ ELBO is relatively loose.
