[toc]
# 1 导言(Introduction)
我们中的大部分人是在学校最后一次见到微积分(calculus),但是导数是机器学习的重要部分，尤其在深度神经网络领域，它通过优化loss function 来训练模型。随便一篇机器学习论文，或者深度学习框架的文档，其中涉及的不仅是数值积分，还有矩阵积分，它是线性代数和多变量微分的结合。  
通过现代化的机器学习框架，你可以在仅仅掌握数值积分的水平下达到世界级深度学习参与者。如果需要理解这些库的底层实现，或者理解一些前言的训练技术，你需要理解矩阵微积分的特定部分。

![xx](./png/01.png)
如上图所示，对于一个神经网络计算单元，$z(\textbf{x}) = \sum_i^n w_i x_i + b = \textbf{w} \cdot \textbf{x} + b$
函数$F(\textbf{x})$称为放射函数(affine function),其后跟着一个[线性修正单元](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))，也即是激活函数;它将负值修改为0:$max(0, z(\textbf{x}))$。
神经网络由这些单元组成，这些单元被组织为$layers$  
训练神经网络也即是通过最小化$loss function$选择合适的$\textbf{w}$和偏置$b$.优化算法包括$SGD, SGD\ with\ momentum, Adam$。这里需要获取$activation(\textbf{x})$相对于$\textbf{w}$和$b$的导数。
例如，均方差损失如下：
$$\frac{1}{N} \sum_{\mathbf{x}}(\operatorname{target}(\mathbf{x})-\operatorname{activation}(\mathbf{x}))^{2}=\frac{1}{N} \sum_{\mathbf{x}}\left(\operatorname{target}(\mathbf{x})-\max \left(0, \sum_{i}^{|x|} w_{i} x_{i}+b\right)\right)^{2}$$
# 2 数值函数求导法则(Scalar derivative rules)
|法则|数学符号|对于$x$的导数|
|:--:|:--:|:--:|
|Constant|$c$|0|
|数乘|$cf$|$c \frac{d f}{d x}$|
|指数|$x^n$|$nx^{n-1}$|
|加法|$f + g$|$\frac{d f}{d x} + \frac{d g}{d x}$|
|减法|$f - g$|$\frac{d f}{d x} - \frac{d g}{d x}$|
|乘法|$f g$|$\frac{d f}{d x}g + \frac{d g}{d x} f$|
|链式|$f(g(x))$|$\frac{d f(u)}{d u} \frac{d u}{d x}$|
# 3 向量积分和偏导数(vector calculus and partial derivatives)
对于多变量函数$f(x, y) = 3 x^2 y$，它的梯度可以由如下向量表示

$$\nabla f(x, y)=\left[\frac{\partial f(x, y)}{\partial x}, \frac{\partial f(x, y)}{\partial y}\right]=\left[6 y x, 3 x^{2}\right]$$

这里处理的是 $\vec{x}$ 向数值 $z$ 的映射，下面的矩阵积分将处理$n$维向$m$维的映射
# 4 矩阵积分(Matrix calculus)
首先引入$g(x, y) = 2 x + y^8$
$$\frac{\partial g(x, y)}{\partial x}=\frac{\partial 2 x}{\partial x}+\frac{\partial y^{8}}{\partial x}=2 \frac{\partial x}{\partial x}+0=2 \times 1=2$$
$$\frac{\partial g(x, y)}{\partial y}=\frac{\partial 2 x}{\partial y}+\frac{\partial y^{8}}{\partial y}=0+8 y^{7}=8 y^{7}$$
对于该函数的梯度表示如下：
$$\nabla g(x, y)=\left[2,8 y^{7}\right]$$
通过将两个函数的梯度叠放到一个矩阵里面，可以得到如下结果：
$$J=\left[\begin{array}{c}
\nabla f(x, y) \\
\nabla g(x, y)
\end{array}\right]=\left[\begin{array}{ll}
\frac{\partial f(x, y)}{\partial x} & \frac{\partial f(x, y)}{\partial y} \\
\frac{\partial g(x, y)}{\partial x} & \frac{\partial g(x, y)}{\partial y}
\end{array}\right]=\left[\begin{array}{cc}
6 y x & 3 x^{2} \\
2 & 8 y^{7}
\end{array}\right]$$
这种放置方法成为**numerator layout**,也有很多文献使用**denominator layout**方法，该方法为**numerator layout**的 Transpose.
## 4.1 Jacobian 的推广
对于多元函数，我们可以将其推广到向量方程
$$f(x, y, z) \Rightarrow f(\mathbf{x})$$
黑体表示向量$\mathbf{x}$, 斜体字为数值 $x$;
假定所有向量为列向量,也即是$n \times 1$:
$$\mathbf{x}=\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right]$$
对于多个标量函数，我们可以将其合并为向量。
$$\mathbf{y}=\mathbf{f}(\mathbf{x})$$
其中$|\mathbf{x}| = n$, $|\mathbf{y}| = m$
类似如下表示：
$$\begin{aligned}
y_{1} &=f_{1}(\mathbf{x}) \\
y_{2} &=f_{2}(\mathbf{x}) \\
& \vdots \\
y_{m} &=f_{m}(\mathbf{x})
\end{aligned}$$
以下是简单的例子
$\mathbf{y} = \mathbf{f}(\mathbf{x}) = \mathbf{x}$,对应的标量函数如下
$$\begin{aligned}
y_1 &= f_1 (\mathbf{x}) = x_1\\
y_2 &= f_2 (\mathbf{x}) = x_2\\
& \vdots \\
y_n &= f_n (\mathbf{x}) = x_n\\
\end{aligned}$$
通常来说Jacobian矩阵包含$m \times n$个可能的偏导数，$m$ 对应标量函数的数量，$n$ 对应输入向量的维度
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
=\left[\begin{array}{c}
\nabla f_{1}(\mathbf{x}) \\
\nabla f_{2}(\mathbf{x}) \\
\ldots \\
\nabla f_{m}(\mathbf{x})
\end{array}\right]
=\left[\begin{array}{c}
\frac{\partial}{\partial \mathbf{x}} f_{1}(\mathbf{x}) \\
\frac{\partial}{\partial \mathbf{x}} f_{2}(\mathbf{x}) \\
\ldots \\
\frac{\partial}{\partial \mathbf{x}} f_{m}(\mathbf{x})
\end{array}\right]
=\left[\begin{array}{ccc}
\frac{\partial}{\partial x_{1}} f_{1}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{1}(\mathbf{x}) & \ldots & \frac{\partial}{\partial x_{n}} f_{1}(\mathbf{x}) \\
\frac{\partial}{\partial x_{1}} f_{2}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{2}(\mathbf{x}) & \ldots & \frac{\partial}{\partial x_{n}} f_{2}(\mathbf{x}) \\
                                                  & \ldots \\
\frac{\partial}{\partial x_{1}} f_{m}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{m}(\mathbf{x}) & \ldots & \frac{\partial}{\partial x_{n}} f_{m}(\mathbf{x})
\end{array}\right]$$
每一个$\frac{\partial}{\partial \mathbf{x}} f_{i}(\mathbf{x})$对应一个水平的向量。  
对于函数$\mathbf{f}(\mathbf{x}) = \mathbf{x}$,也即是 $f_i(\mathbf{x}) = x_i$, 其对应的Jacobian矩阵如下：
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
=\left[\begin{array}{c}
\frac{\partial}{\partial \mathbf{x}} f_{1}(\mathbf{x}) \\
\frac{\partial}{\partial \mathbf{x}} f_{2}(\mathbf{x}) \\
\ldots \\
\frac{\partial}{\partial \mathbf{x}} f_{m}(\mathbf{x})
\end{array}\right]
=\left[\begin{array}{cccc}
\frac{\partial}{\partial x_{1}} f_{1}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{1}(\mathbf{x}) & \dots & \frac{\partial}{\partial x_{n}} f_{1}(\mathbf{x}) \\
\frac{\partial}{\partial x_{1}} f_{2}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{2}(\mathbf{x}) & \dots & \frac{\partial}{\partial x_{n}} f_{2}(\mathbf{x}) \\
& \cdots & & \\
\frac{\partial}{\partial x_{1}} f_{m}(\mathbf{x}) & \frac{\partial}{\partial x_{2}} f_{m}(\mathbf{x}) & \dots & \frac{\partial}{\partial x_{n}} f_{m}(\mathbf{x})
\end{array}\right]$$
$$
=\left[\begin{array}{cccc}
\frac{\partial}{\partial x_{1}} x_{1} & \frac{\partial}{\partial x_{2}} x_{1} & \dots & \frac{\partial}{\partial x_{n}} x_{1} \\
\frac{\partial}{\partial x_{1}} x_{2} & \frac{\partial}{\partial x_{2}} x_{2} & \dots & \frac{\partial}{\partial x_{n}} x_{2} \\
&\cdots & & \\
\frac{\partial}{\partial x_{1}} x_{n} & \frac{\partial}{\partial x_{2}} x_{n} & \dots & \frac{\partial}{\partial x_{n}} x_{n}
\end{array}\right]
=\left[\begin{array}{cccc}
\frac{\partial}{\partial x_{1}} x_{1} & 0 & \dots & 0 \\
0 & \frac{\partial}{\partial x_{2}} x_{2} & \dots & 0 \\
& & \ddots & \\
0 & 0 & \dots & \frac{\partial}{\partial x_{n}} x_{n}
\end{array}\right]
=\left[
    \begin{array}{cccc}
    1 & 0 & \cdots & 0 \\
    0 & 1 & \cdots & 0 \\
      &   & \ddots & \\
    0 & 0 & \cdots & 1
    \end{array}
\right] 
 = \mathbf{I}$$

 ## 4.2 element-wise 二元操作
 element-wise 二元操作是类似向量逐个元素相加。例如 $max(\textbf{w}, \textbf{x})$或者$\textbf{w} > \textbf{x}$.
当然，我们也可以推广元素级别的操作，使用如下符号表示$\textbf{y} = \textbf{f}(\textbf{w}) \bigcirc \textbf{g}(\textbf{x})$, 这也意味着 输出向量$\textbf{y}$ 同输入向量$\textbf{x}$ 维度相同均为$n$.  
展开如下
$$\left[\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{n}
\end{array}\right]
=\left[\begin{array}{c}
f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x}) \\
f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x}) \\
\vdots \\
f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})
\end{array}\right]$$


关于w的Jacobian矩阵如下
$$J_{\mathrm{w}}=\frac{\partial \mathbf{y}}{\partial \mathbf{w}}
=\left[\begin{array}{llll}
\frac{\partial}{\partial w_{1}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) & \frac{\partial}{\partial w_{2}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) & \dots & \frac{\partial}{\partial w_{n}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) \\
\frac{\partial}{\partial w_{1}} \left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x}) \right) & \frac{\partial}{\partial w_{2}}\left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x})\right) & \dots & \frac{\partial}{\partial w_{n}}\left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x})\right) \\
& \dots \\
\frac{\partial}{\partial w_{1}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right) & \frac{\partial}{\partial w_{2}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right) & \cdots & \frac{\partial}{\partial w_{n}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right)
\end{array}\right]$$
类似，可以得到关于$\textbf{x}$的矩阵:
$$J_{\mathrm{x}}=\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
=\left[\begin{array}{llll}
\frac{\partial}{\partial x_{1}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) & \frac{\partial}{\partial x_{2}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) & \dots & \frac{\partial}{\partial x_{n}}\left(f_{1}(\mathbf{w}) \bigcirc g_{1}(\mathbf{x})\right) \\
\frac{\partial}{\partial x_{1}} \left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x}) \right) & \frac{\partial}{\partial x_{2}}\left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x})\right) & \dots & \frac{\partial}{\partial x_{n}}\left(f_{2}(\mathbf{w}) \bigcirc g_{2}(\mathbf{x})\right) \\
& \dots \\
\frac{\partial}{\partial x_{1}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right) & \frac{\partial}{\partial x_{2}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right) & \cdots & \frac{\partial}{\partial x_{n}}\left(f_{n}(\mathbf{w}) \bigcirc g_{n}(\mathbf{x})\right)
\end{array}\right]$$
由于element-wise 操作的性质，可以得到$\hat{f}_{i}(w_{i})=f_{i}(\mathbf{w})$，也即是 $\frac{\partial} {\partial x_i} (f_{k}(\mathbf{w}) \bigcirc f_{k}(\mathbf{x})) = \frac{\partial}{\partial x_i} (f_{k}(w_j) \bigcirc f_{k}(x_j))$, 由导数相关法则可知, 若 $i \not = j$， 则 $\frac{\partial}{\partial x_i} (f_{k}(w_j) \bigcirc f_{k}(x_j)) = 0$, 上述公式可以简化为如下:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{w}}
=\left[\begin{array}{ccc}
\frac{\partial}{\partial w_{1}}(f_{1}(w_{1}) \bigcirc g_{1}(x_{1})) & & & \Huge0\\
& \frac{\partial}{\partial w_{2}}  (f_{2}(w_{2}) \bigcirc g_{2}(x_{2})) & & \\
& &  \ldots & &\\
\Huge0 & &  & \frac{\partial} {\partial x_{n}} (f_{n}(w_{n}) \bigcirc g_{n}(x_{n}) )
\end{array}\right]$$
也可以使用如下的简洁表示方式
$$\frac{\partial \mathbf{y}}{\partial \mathbf{w}}=\operatorname{diag}\left(\frac{\partial}{\partial w_{1}}\left(f_{1}\left(w_{1}\right) \bigcirc g_{1}\left(x_{1}\right)\right), \frac{\partial}{\partial w_{2}}\left(f_{2}\left(w_{2}\right) \bigcirc g_{2}\left(x_{2}\right)\right), \ldots, \frac{\partial}{\partial w_{n}}\left(f_{n}\left(w_{n}\right) \bigcirc g_{n}\left(x_{n}\right)\right)\right)$$
关于$\mathbf{x}$可以有类似表达
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\operatorname{diag}\left(\frac{\partial}{\partial x_{1}}\left(f_{1}\left(w_{1}\right) \bigcirc g_{1}\left(x_{1}\right)\right), \frac{\partial}{\partial x_{2}}\left(f_{2}\left(w_{2}\right) \bigcirc g_{2}\left(x_{2}\right)\right), \ldots, \frac{\partial}{\partial x_{n}}\left(f_{n}\left(w_{n}\right) \bigcirc g_{n}\left(x_{n}\right)\right)\right)$$

## 4.3涉及标量的导数

依据之前的推论：
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\operatorname{diag}\left(\ldots \frac{\partial}{\partial x_{i}}\left(f_{i}\left(x_{i}\right) \bigcirc g_{i}(z)\right) \ldots\right)$$
假设$\mathbf{f}(\mathbf{x}) = \mathbf{x}$, $\mathbf{g}(z)=\overrightarrow{1} z$

例如， 对于加法： $\mathbf{y} = \mathbf{x} + z$   
由：
$$\frac{\partial}{\partial x_{i}}\left(f_{i}\left(x_{i}\right)+g_{i}(z)\right)=\frac{\partial\left(x_{i}+z\right)}{\partial x_{i}}=\frac{\partial x_{i}}{\partial x_{i}}+\frac{\partial z}{\partial x_{i}}=1+0=1$$
可得：
$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x} + z) = diag(\overrightarrow{{1}}) = \mathbf{I}$$
对于标量$z$, 可得如下结果
$$\frac{\partial}{\partial z}(f_i(x_i) + g_i(z)) = \frac{\partial(x_i + z)}{\partial z} = \frac{\partial x_i}{\partial z} + \frac{\partial z}{\partial z} = 0 + 1 = 1$$
所以
$$\frac{\partial}{\partial z}(\mathbf{x} + z) = \overrightarrow{1}$$


对于乘法： $\mathbf{y} = \mathbf{x} z$ 
关于$\mathbf{x}$的导数
$$\frac{\partial}{\partial x_{i}}\left(f_{i}\left(x_{i}\right) \otimes g_{i}(z)\right)=x_{i} \frac{\partial z}{\partial x_{i}}+z \frac{\partial x_{i}}{\partial x_{i}}=0+z=z$$
所以:
$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x} z)=\operatorname{diag}(\overrightarrow{\mathbf{1}} z)=I z$$
关于$z$的导数
$$\frac{\partial}{\partial z}\left(f_{i}\left(x_{i}\right) \otimes g_{i}(z)\right)=x_{i} \frac{\partial z}{\partial z}+z \frac{\partial x_{i}}{\partial z}=x_{i}+0=x_{i}$$
所以：
$$\frac{\partial}{\partial z}(\mathbf{x} z)=\mathbf{x}$$

## 4.4 向量求和导数

假设 $ y = sum(\mathbf{f}(\mathbf{x})) = \sum_{i=1}^n f_i(\mathbf{x})$.
对于$\mathbf{x}$的导数如下：
$$\begin{aligned}
\frac{\partial y}{\partial \mathbf{x}} &=\left[\frac{\partial y}{\partial x_{1}}, \frac{\partial y}{\partial x_{2}}, \ldots, \frac{\partial y}{\partial x_{n}}\right] \\
&=\left[\frac{\partial}{\partial x_{1}} \sum_{i} f_{i}(\mathbf{x}), \frac{\partial}{\partial x_{2}} \sum_{i} f_{i}(\mathbf{x}), \ldots, \frac{\partial}{\partial x_{n}} \sum_{i} f_{i}(\mathbf{x})\right] \\
&=\left[\sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{1}}, \sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{2}}, \ldots, \sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{n}}\right]
\end{aligned}$$

例如，对于$y = sum(\mathbf{x})$, $f_i(\mathbf{x}) = x_i$,其导数如下：
$$\nabla y=\left[\sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{1}}, \sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{2}}, \ldots, \sum_{i} \frac{\partial f_{i}(\mathbf{x})}{\partial x_{n}}\right]=\left[\sum_{i} \frac{\partial x_{i}}{\partial x_{1}}, \sum_{i} \frac{\partial x_{i}}{\partial x_{2}}, \ldots, \sum_{i} \frac{\partial x_{i}}{\partial x_{n}}\right]$$
又因为，$\frac{\partial}{\partial x_j} x_i = 0$，对于$j \not = i$
所以：
$$\nabla y=\left[\begin{array}{lll}
\frac{\partial x_{1}}{\partial x_{1}}, & \frac{\partial x_{2}}{\partial x_{2}}, & \ldots,
\frac{\partial x_{n}}{\partial x_{n}}\end{array}\right]=[1,1, \ldots, 1]=\overrightarrow{1}^{T}$$

## 4.5 链式法则(Chain Rules)
### 4.5.1 单变量链式法则
对于 $y = f(g(x))$， 链式法则如下：
$$\frac{d y}{d x} = \frac{d y}{d u} \frac{d u}{d x},\  \ \  u = g(x)$$
- forward differentiation  参数 如何影响 函数输出
$$\frac{d y}{d x} = \frac{d u}{d x} \frac{d y}{d u} ,\  \ \  u = g(x)$$
- backward differentiation: 函数输出 如何影响 参数
$$\frac{d y}{d x} = \frac{d y}{d u} \frac{d u}{d x},\  \ \  u = g(x)$$
- **单变量链式法则适用场景**
  - 参数$x$到输出$y$只有一条数据流路径, 例如对于$y = sin(x^2)$
![xx](./png/02.png)
### 4.5.2 单变量全微分链式法则
需要考虑$x$变化影响输出$y$的所有路径, 例如对于$y = x + x^2$
![xx](./png/03.png)
对于$y = x + x^2$我们可以视其为
$$\begin{aligned}
u_1(x) &=x^2 &\\
u_2(x, u_1) &= x + u_1 &\left(y = f(x) = u_2(x, u_1)\right)
\end{aligned}$$
$y$ 对应 $x$ 的全微分为
$$\frac{d y}{d x} = \frac{\partial f(x)}{\partial x} = \frac{\partial u_2}{\partial x} + \frac{\partial u_2}{\partial u_1} \frac{\partial u_1}{\partial x} = 1 + 2x$$
推广公式如下：
$$\frac{\partial f(x, u_1, ..., u_n)}{\partial x} = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial u_1} \frac{\partial u_1}{\partial x} + \frac{\partial f}{\partial u_1} \frac{\partial u_1}{\partial x} + \ldots + \frac{\partial f}{\partial u_n} \frac{\partial u_n}{\partial x} = \frac{\partial f}{\partial x} + \sum_{i=1}^n \frac{f}{\partial u_i} \frac{\partial u_i}{\partial x}$$
令$u_{n+1} = x$
$$\frac{\partial f(u_1, ..., u_{n+1})}{\partial x} =  \sum_{i=1}^{n+1} \frac{f}{\partial u_i} \frac{\partial u_i}{\partial x}$$
这里可以看出类似,向量总和的形式$\frac{\partial f}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial x}$
### 4.5.3 向量链式法则
$\mathbf{y} = \mathbf{f}(x)$:
$$\left[\begin{array}{l}
y_{1}(x) \\
y_{2}(x)
\end{array}\right]
=\left[\begin{array}{l}
f_{1}(x) \\
f_{2}(x)
\end{array}\right]
=\left[\begin{array}{l}
\ln \left(x^{2}\right) \\
\sin (3 x)
\end{array}\right]$$
引入中间变量，$\mathbf{y} = \mathbf{f}(\mathbf{g}(x))$:
$$\left[\begin{array}{l}
g_{1}(x) \\
g_{2}(x)
\end{array}\right]
=\left[\begin{array}{l}
x^{2} \\
3 x
\end{array}\right]$$
$$
\left[\begin{array}{l}
f_{1}(\mathbf{g}) \\
f_{2}(\mathbf{g})
\end{array}\right]=\left[\begin{array}{l}
\ln \left(g_{1}\right) \\
\sin \left(g_{2}\right)
\end{array}\right]$$
$\mathbf{y}$对于$x$的导数如下：
$$\frac{\partial \mathbf{y}}{\partial x}=\left[\begin{array}{l}
\frac{\partial f_{1}(\mathbf{g})}{\partial x_{1}} \\
\frac{\partial f_{2}(\mathbf{g})}{\partial x}
\end{array}\right]=\left[\begin{array}{l}
\frac{\partial f_{1}}{\partial g_{1}} \frac{\partial g_{1}}{\partial x}+\frac{\partial f_{1}}{\partial g_{2}} \frac{\partial g_{2}}{\partial x_{2}} \\
\frac{\partial f_{2}}{\partial g_{1}} \frac{\partial_{1}}{\partial x}+\frac{\partial f_{2}}{\partial g_{2}} \frac{1}{\partial x}
\end{array}\right]=\left[\begin{array}{c}
\frac{1}{g_{1}} 2 x+0 \\
0+\cos \left(g_{2}\right) 3
\end{array}\right]=\left[\begin{array}{c}
\frac{2 x}{x^{2}} \\
3 \cos (3 x)
\end{array}\right]=\left[\begin{array}{c}
\frac{2}{x} \\
3 \cos (3 x)
\end{array}\right]$$
可以作如下变换
$$\left[\begin{array}{l}
\frac{\partial f_{1}}{\partial g_{1}} \frac{\partial g_{1}}{\partial x}+\frac{\partial f_{1}}{\partial g_{2}} \frac{\partial g_{2}}{\partial x} \\
\frac{\partial f_{2}}{\partial g_{1}} \frac{\partial g_{1}}{\partial x}+\frac{\partial f_{2}}{\partial g_{2}} \frac{\partial g_{2}}{\partial x}
\end{array}\right]
=\left[\begin{array}{ll}
\frac{\partial f_{1}}{\partial g_{1}} & \frac{\partial f_{1}}{\partial g_{2}} \\
\frac{\partial f_{2}}{\partial g_{1}} & \frac{\partial f_{2}}{\partial g_{2}}
\end{array}\right]\left[\begin{array}{l}
\frac{\partial g_{1}}{\partial x} \\
\frac{\partial g_{2}}{\partial x}
\end{array}\right]=\frac{\partial \mathbf{f}}{\partial \mathbf{g}} \frac{\partial \mathbf{g}}{\partial x}$$
也即是
$$\frac{\partial}{\partial x} \mathbf{f}(\mathbf{g}(x))=\frac{\partial \mathbf{f}}{\partial \mathbf{g}} \frac{\partial \mathbf{g}}{\partial x}$$
对于参数为向量的情况：
$$\frac{\partial}{\partial \mathbf{x}} \mathbf{f}(\mathbf{g}(\mathbf{x}))=\frac{\partial \mathbf{f}}{\partial \mathbf{g}} \frac{\partial \mathbf{g}}{\partial \mathbf{x}}$$
展开为矩阵形式如下：
$$\frac{\partial}{\partial \mathbf{x}} \mathbf{f}(\mathbf{g}(\mathbf{x}))=\left[\begin{array}{cccc}
\frac{\partial f_{1}}{\partial g_{1}} & \frac{\partial f_{1}}{\partial g_{2}} & \dots & \frac{\partial f_{1}}{\partial g_{k}} \\
\frac{\partial f_{2}}{\partial g_{1}} & \frac{\partial f_{2}}{\partial g_{2}} & \dots & \frac{\partial f_{2}}{\partial g_{k}} \\
\frac{\partial f_{m}}{\partial g_{1}} & \frac{\partial f_{m}}{\partial g_{2}} & \cdots & \frac{\partial f_{m}}{\partial g_{k}}
\end{array}\right]\left[\begin{array}{cccc}
\frac{\partial g_{1}}{\partial x_{1}} & \frac{\partial g_{1}}{\partial x_{2}} & \cdots & \frac{\partial g_{1}}{\partial x_{n}} \\
\frac{\partial g_{2}}{\partial x_{1}} & \frac{\partial g_{2}}{\partial x_{2}} & \cdots & \frac{\partial g_{2}}{\partial x_{n}} \\
\frac{\partial g_{k}}{\partial x_{1}} & \frac{\partial g_{k}}{\partial x_{2}} & \cdots & \frac{\partial g_{k}}{\partial x_{n}}
\end{array}\right]$$
# 5 神经元激活函数
这里并没有难理解的地方，主要是前面element-wise微分， 向量微分的应用， 还有分段函数微分，**里面提到的广播机制我没觉得有什么用**。
神经元函数表达式
$$activation(\mathbf{x}) = max(0, \mathbf{w} \cdot \mathbf{x} + b)$$
引入中间变量得到如下表达：
$$z(\mathbf{w}, b, \mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$
$$activation(z) = max(0, z)$$
对于函数$max(0, z)$,我们得到如下分段积分
$$\frac{\partial}{\partial z} \max (0, z)=\left\{
\begin{array}{ll}
0 & z \leq 0 \\
\frac{d z}{d z}=1 & z>0
\end{array}\right.$$
根据链式法则
$$\frac{\partial activation}{\partial \mathbf{w}} = \frac{\partial activation}{\partial z}\frac{\partial z}{\partial \mathbf{w}}$$
对于$\frac{\partial activation}{\partial z}$
$$\frac{\partial activation}{\partial z} = \left\{
\begin{array}{ll}
0 & z \leq 0 \\
\frac{d z}{d z}=1 &z > 0
\end{array}
\right.$$
对于$\frac{\partial z}{\partial \mathbf{w}}$
$$\frac{\partial z}{\partial \mathbf{w}} = \frac{\partial }{\partial \mathbf{w}} \mathbf{w} \cdot \mathbf{x} + \frac{\partial}{\partial \mathbf{w}}b = \frac{\partial }{\partial \mathbf{w}} \mathbf{w} \cdot \mathbf{x} + \overrightarrow{0}^T$$
对于$\frac{\partial}{\partial \mathbf{w}}\mathbf{w} \cdot \mathbf{x}$，设$y = \mathbf{w} \cdot \mathbf{x}$引入中间变量得到如下结果
$$\begin{aligned} \mathbf{u} &= \mathbf{w} \cdot \mathbf{x}\\
y& = sum(\mathbf{u})\end{aligned}$$
$$\begin{aligned}
\frac{\partial \mathbf{u}}{\partial \mathbf{w}} &= \frac{\partial}{\partial \mathbf{w}}(\mathbf{w} \cdot \mathbf{x}) = diag(\mathbf{x})\\
\frac{\partial y}{\partial \mathbf{u}} &= \frac{\partial}{\partial \mathbf{u}}sum(\mathbf{u}) =\overrightarrow{1}^T
\end{aligned}$$
所以可得如下结果：
$$\frac{\partial y}{\partial \mathbf{w}} = \frac{\partial y}{\partial \mathbf{u}} \frac{\partial \mathbf{u}}{\partial \mathbf{w}} = \overrightarrow{1}^T diag(\mathbf{x}) = \mathbf{x}^T$$
所以
$$\frac{\partial z}{\partial \mathbf{w}} = \frac{\partial}{\partial \mathbf{w}} \mathbf{w} \cdot \mathbf{x} + \frac{\partial}{\partial \mathbf{w}}b = \mathbf{x}^T + \overrightarrow{0}^T = \mathbf{x}^T  $$
所以
$$\frac{\partial activaion}{\partial \mathbf{w}} = \left\{\begin{array}{ll} 0 \frac{\partial z}{\partial \mathbf{w}} &z \leq 0\\
1 \frac{\partial z}{\partial \mathbf{w}} &z > 0
\end{array}
\right.$$
也即是
$$\frac{\partial activaion}{\partial \mathbf{w}} = \left\{\begin{array}{ll} \overrightarrow{0}^T &\mathbf{w} \cdot \mathbf{x} + b \leq 0\\
 \mathbf{x}^T &\mathbf{w} \cdot \mathbf{x} + b > 0
\end{array}
\right.$$
同样的，对于$b$
$$\frac{\partial activation}{\partial b} = \frac{\partial  activation}{\partial z} \frac{\partial z}{\partial b}$$
$$\frac{\partial z}{\partial b}=\frac{\partial}{\partial b} \mathbf{w} \cdot \mathbf{x}+\frac{\partial}{\partial b} b \quad=\quad 0+1 \quad=1$$
$$\frac{\partial activaion}{\partial b} = \left\{\begin{array}{ll} \overrightarrow{0}^T &\mathbf{w} \cdot \mathbf{x} + b \leq 0\\
 \mathbf{x}^T &\mathbf{w} \cdot \mathbf{x} + b > 0
\end{array}
\right.$$
所以
$$\frac{\partial activaion}{\partial b} = \left\{\begin{array}{ll} 0 &\mathbf{w} \cdot \mathbf{x} + b \leq 0\\
 1 &\mathbf{w} \cdot \mathbf{x} + b > 0
\end{array}
\right.$$
# 6 神经网络损失函数的梯度
假设神经网络输入如下
$$\mathbf{x} = \left[ \mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_N \right]^T$$
标签如下
$$\mathbf{y}=\left[\operatorname{target}\left(\mathbf{x}_{1}\right), \operatorname{target}\left(\mathbf{x}_{2}\right), \ldots, \operatorname{target}\left(\mathbf{x}_{N}\right)\right]^{T} = \left[y_1, y_2, \cdots, y_N\right]$$
损失函数
$$C(\mathbf{w}, b, X, \mathbf{y})=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\operatorname{activation}\left(\mathbf{x}_{i}\right)\right)^{2}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\max \left(0, \mathbf{w} \cdot \mathbf{x}_{i}+b\right)\right)^{2}$$
定义中间变量如下
$$
\begin{array}{l}
u(\mathbf{w}, b, \mathbf{x}) \quad=\quad \max (0, \mathbf{w} \cdot \mathbf{x}+b) \\
v(y, u) \quad=\quad y-u \\
C(v) \quad=\quad \frac{1}{N} \sum_{i=1}^{N} v^{2}
\end{array}
$$
## 6.1 对于参数 $\mathbf{w}$ 的梯度
因为
$$\frac{\partial}{\partial \mathbf{w}} u(\mathbf{w}, b, \mathbf{x})=\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
\mathbf{x}^{T} & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.$$
所以
$$\frac{\partial v(y, u)}{\partial \mathbf{w}}=\frac{\partial}{\partial \mathbf{w}}(y-u)=\overrightarrow{0}^{T}-\frac{\partial u}{\partial \mathbf{w}}=-\frac{\partial u}{\partial \mathbf{w}}=\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
-\mathbf{x}^{T} & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.$$
所以
$$\begin{aligned}
\frac{\partial C(v)}{\partial \mathbf{w}} &=\frac{\partial}{\partial \mathbf{w}} \frac{1}{N} \sum_{i=1}^{N} v^{2} \\
&=\frac{1}{N} \sum_{i=1}^{N} \frac{\partial}{\partial \mathbf{w}} v^{2} \\
&=\frac{1}{N} \sum_{i=1}^{N} \frac{\partial v^{2}}{\partial v} \frac{\partial v}{\partial \mathbf{w}} \\
&=\frac{1}{N} \sum_{i=1}^{N} 2 v \frac{\partial v}{\partial \mathbf{w}} \\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
2 v \overrightarrow{0}^{T}=\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
-2 v \mathbf{x}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.\\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
-2\left(y_{i}-u\right) \mathbf{x}_{i}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.\\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
-2\left(y_{i}-\max \left(0, \mathbf{w} \cdot \mathbf{x}_{i}+b\right)\right) \mathbf{x}_{i}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.\\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
-2\left(y_{i}-\left(\mathbf{w} \cdot \mathbf{x}_{i}+b\right)\right) \mathbf{x}_{i}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.\\
&=\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
\frac{-2}{N} \sum_{i=1}^{N}\left(y_{i}-\left(\mathbf{w} \cdot \mathbf{x}_{i}+b\right)\right) \mathbf{x}_{i}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.\\
&=\left\{\begin{array}{ll}
\overrightarrow{0}^{T} & \text { w } \cdot \mathbf{x}_{i}+b \leq 0 \\
\frac{2}{N} \sum_{i=1}^{N}\left(\mathbf{w} \cdot \mathbf{x}_{i}+b-y_{i}\right) \mathbf{x}_{i}^{T} & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.
\end{aligned}$$
令 $e_i = \mathbf{w} \cdot \mathbf{x}_i + b - y_i$
则
$$\frac{\partial C}{\partial \mathbf{w}} = \frac{2}{N} \sum_{i=1}^N e_i \mathbf{x}_i^T$$
假定输入向量只有一个，损失值为$2e_1\mathbf{x}_1^T$.如果错误$e_1$为0,那么损失值为0; 如果$e_1$为正数，那么梯度方向在$\mathbf{x}_1$方向，如果$e_1$为负值，那么梯度方向为$\mathbf{x}_1$的负方向
对于梯度下降算法，我们需要向梯度负方向移动：
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\partial C}{\partial \mathbf{w}}$$
## 6.2 对于偏置$b$的微分
$$\begin{aligned}
u(\mathbf{w}, b, \mathbf{x}) &=\max (0, \mathbf{w} \cdot \mathbf{x}+b) \\
v(y, u) &=y-u \\
C(v) &=\frac{1}{N} \sum_{i=1}^{N} v^{2}
\end{aligned}$$
对于函数$u$：
$$
\frac{\partial u}{\partial b}=\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
1 & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.
$$
对于函数$v$:
$$
\frac{\partial v(y, u)}{\partial b}=\frac{\partial}{\partial b}(y-u)=0-\frac{\partial u}{\partial b}=-\frac{\partial u}{\partial b}=\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
-1 & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.
$$
对于损失函数：
$$\begin{aligned}
\frac{\partial C(v)}{\partial b} &=\frac{\partial}{\partial b} \frac{1}{N} \sum_{i=1}^{N} v^{2} \\
&=\frac{1}{N} \sum_{i=1}^{N} \frac{\partial}{\partial b} v^{2} \\
&=\frac{1}{N} \sum_{i=1}^{N} \frac{\partial v^{2}}{\partial v} \frac{\partial v}{\partial b} \\
&=\frac{1}{N} \sum_{i=1}^{N} 2 v \frac{\partial v}{\partial b} \\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
-2 v & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.\\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
-2\left(y_{i}-\max \left(0, \mathbf{w} \cdot \mathbf{x}_{i}+b\right)\right) & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.\\
&=\frac{1}{N} \sum_{i=1}^{N}\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}+b \leq 0 \\
2\left(\mathbf{w} \cdot \mathbf{x}_{i}+b-y_{i}\right) & \mathbf{w} \cdot \mathbf{x}+b>0
\end{array}\right.\\
&=\left\{\begin{array}{ll}
0 & \mathbf{w} \cdot \mathbf{x}_{i}+b \leq 0 \\
\frac{2}{N} \sum_{i=1}^{N}\left(\mathbf{w} \cdot \mathbf{x}_{i}+b-y_{i}\right) & \mathbf{w} \cdot \mathbf{x}_{i}+b>0
\end{array}\right.
\end{aligned}$$
与之前类似
$$\frac{\partial C}{\partial b}=\frac{2}{N} \sum_{i=1}^{N} e_{i} $$
参数优化方式如下
$$
b_{t+1}=b_{t}-\eta \frac{\partial C}{\partial b}
$$
# 矩阵求导公式参考
[wiki百科](https://en.wikipedia.org/wiki/Matrix_calculus)