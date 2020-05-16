import numpy as np 
import matplotlib.pyplot as plt

def gen_data(num=100, mu=10, var=2, label=1):
    ret = list()
    labels = list()
    for _ in range(num):
        #tmp = [np.random.normal(loc=mu, scale=var), np.random.normal(loc=mu, scale=var)]
        # 加入噪声的数据
        tmp = [np.random.normal(loc=mu, scale=var) + np.random.normal(0, 1), np.random.normal(loc=mu, scale=var) + np.random.normal(0, 1)]
        ret.append(tmp)
        labels.append(label)
    return ret, labels


def plt_picture(data, color="red"):
    x = list()
    y = list()
    for it in data:
        x.append(it[0])
        y.append(it[1])

    plt.scatter(x, y, c=color)

def normal_pdf(x, mu, var):
    return 1 / np.math.sqrt(2 * np.math.pi* var) * np.math.exp(-(x - mu) * (x - mu) / (2 * var))


class NB:
    def __init__(self):
        pass

    # 这里假设，特征都是高斯分布的, x_1 代表特征1， x_2 代表特征2, y表示标签
    def evaluate_para(self, p_data, n_data):
        p_mu = np.mean(p_data, axis=0)
        p_cov = np.cov(p_data.T)
        n_mu = np.mean(n_data, axis=0)
        n_cov = np.cov(n_data.T)

        # 这里计算的时p(x_1| y = positive), p(x_2| y = positive)的概率分布的参数
        self.p_f_1_mu = p_mu[0]
        self.p_f_1_var = p_cov[0][0]
        self.p_f_2_mu = p_mu[1]
        self.p_f_2_var = p_cov[1][1]
        
        # 这里计算的时p(x_1| y = negative), p(x_2| y = negative)的概率分布的参数
        self.n_f_1_mu = n_mu[0]
        self.n_f_1_var = n_cov[0][0]
        self.n_f_2_mu = n_mu[1]
        self.n_f_2_var = n_cov[1][1]

        # 这里计算的是正负类的先验分布
        self.p_y = p_data.shape[0] / (p_data.shape[0] + n_data.shape[0])
        self.n_y = n_data.shape[0] / (p_data.shape[0] + n_data.shape[0])

    def predict(self, x):
        # 为正类的概率
        p_prob = self.p_y * normal_pdf(x[0], self.p_f_1_mu, self.p_f_1_var) * normal_pdf(x[1], self.p_f_2_mu, self.p_f_2_var)
        # 为负类的概率
        n_prob = self.n_y * normal_pdf(x[0], self.n_f_1_mu, self.n_f_1_var) * normal_pdf(x[1], self.n_f_2_mu, self.n_f_2_var)
        if p_prob > n_prob:
            return 1
        else:
            return -1
        

if __name__ == "__main__":
    """
    准备阶段: 首先生成两个独立的高斯分布数据，一个作为正样本，一个作为负样本,对于 数据集 data,正样本使用标签1， 负样本使用标签-1；
    建模阶段：
    (1) 假设 正负样本分别服从高斯分布，依据样本，计算 均值，方差；
    (2) 依据高斯分布概率密度函数，可以计算得到每个特征的概率分布p(x_1|y = 1),p(x_2|y = 1), p(x_1|y=-1),p(x_2|y=-1)
    (3) 依据数据可以计算y的先验分布p(y=1), p(y=-1)
    (4) 根据prediction 公式 p(y|x) 正比于 p(x_1|y)p(x_2|y)p(y)
    (5) 对比 p(y=1|x), p(y=-1|x) 的大小，获取预测结果
    
    实践心得：
    朴素贝叶斯比较简单，没有太多的难点，主要是p(x_1|y)的分布如何表示，《统计机器学习》那本书中使用的离散分布，在计算机中可以用数组表示；
    这里对于连续型分布反而简单一些，可以直接使用pdf计算对应的值，便于实现。
    存在的问题:
    这是使用的是连续分布的概率密度函数p(x|y)计算后续的正类，负类的后验，按照微积分的想法的话
    应该与离散分布效果一致。
    """
    p_data, _ = gen_data(1000, mu=10, label=1)
    n_data, _ = gen_data(3000, mu=5, label=-1)
    plt_picture(p_data, color="red")
    plt_picture(n_data, color="blue")
    plt.show()
    p_data = np.array(p_data)
    n_data = np.array(n_data)
    nb = NB()
    nb.evaluate_para(p_data, n_data)
    p_data, p_label = gen_data(100, mu=10, label=1)
    n_data, n_label = gen_data(50, mu=5, label=-1)
    right = 0
    all = len(p_label + n_label)
    for it, label in zip((p_data + n_data), (p_label + n_label)):
        if nb.predict(it) == label:
            right += 1
    print(right / all)

    