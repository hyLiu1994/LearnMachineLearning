import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plane_function(x, y):
    return x + 2 * y + 4 

def data_generator(num):
    #x [-100, 100]
    #y [-100, 100]
    # x + 2y + 4 = 0
    datas = list()
    labels = list()
    for _ in range(num):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        v = plane_function(x, y)
        if (v < 0 and v > -20) or (v > 0 and v < 20):
            continue
        datas.append([x, y])
        labels.append(1 if plane_function(x, y) > 0 else 0)
    datas = np.array(datas).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    labels = labels.reshape([labels.shape[0], 1])
    return np.array(datas), np.array(labels, dtype=np.float32)

def draw_picture(datas):
    p_samples = [it for it in datas if plane_function(it[0], it[1]) > 0]
    p_samples_x = [x for x, y in p_samples]
    p_samples_y = [y for x, y in p_samples]

    n_samples = [it for it in datas if plane_function(it[0], it[1]) < 0]
    n_samples_x = [x for x, y in n_samples]
    n_samples_y = [y for x, y in n_samples]

    plt.scatter(p_samples_x, p_samples_y, c="red")
    plt.scatter(n_samples_x, n_samples_y, c="blue")

class LR(tf.keras.Model):
    def __init__(self):
        super(LR, self).__init__()
        self.w = tf.Variable(tf.random.normal([2, 1]), name="w")

    def call(self, x):
        wx = tf.matmul(x, self.w)
        p = tf.keras.activations.sigmoid(wx)
        return p

@tf.function
def train_step(datas, labels, model):
    with tf.GradientTape() as tape:
        pred = model(datas)
        loss = tf.keras.losses.BinaryCrossentropy()(labels, pred)
    grad = tape.gradient(loss, lr.trainable_variables)
    opti.apply_gradients(zip(grad, lr.trainable_variables))
    return loss

def cosine_sim(x, y):
    x_l2_norm = tf.linalg.norm(x)
    y_l2_norm = tf.linalg.norm(y)
    dot_xy = tf.matmul(tf.transpose(x), y)
    cosine = tf.divide(dot_xy, x_l2_norm * y_l2_norm)
    cosine = tf.squeeze(cosine)
    return cosine

if __name__ == "__main__":
    """
    这里使用的随机数据生成方式与前面的相同，通过模拟发现：
    逻辑回归寻找的向量，与之前寻找的向量相同。
    """
    datas, labels = data_generator(100)
    lr = LR()
    opti = tf.keras.optimizers.Adam(0.001)
    epoches = 40000
    w = tf.constant([[1], [2]], dtype=tf.float32)
    cosine_list = list()
    loss_list = list()
    for epoch in range(epoches):
        loss = train_step(datas, labels, lr)
        if epoch % 1000 == 0:
            cosine = cosine_sim(w, lr.w)
            cosine_list.append(cosine.numpy())
            loss_list.append(loss)
            print("epoch: {} loss: {} cosine: {}".format(epoch, loss, cosine))
    
    x = [i for i in range(len(cosine_list))]
    plt.plot(x, cosine_list, c="red")
    plt.show()
    plt.plot(x, loss_list, c="gray")
    plt.show()
    