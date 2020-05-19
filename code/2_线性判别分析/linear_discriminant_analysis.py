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
        if plane_function(x, y) == 0:
            continue
        datas.append([x, y])
        labels.append(1 if plane_function(x, y) > 0 else -1)
    datas = np.array(datas)
    labels = np.array(labels)
    labels = labels.reshape([labels.shape[0], 1])
    return np.array(datas), np.array(labels, dtype=np.float32)

def process_data(datas, labels):
    p_samples_x = list()
    p_samples_y = list()
    n_samples_x = list()
    n_samples_y = list()
    for x, y in zip(datas, labels):
        if y > 0 :
            p_samples_x.append(x)
            p_samples_y.append(y)
        else:
            n_samples_x.append(x)
            n_samples_y.append(y)
    
    p_avg_x = np.mean(p_samples_x, 0)
    n_avg_x = np.mean(n_samples_x, 0)
    p_samples_x = p_samples_x - p_avg_x
    n_samples_x = n_samples_x - n_avg_x
    p_var_x = np.matmul(np.transpose(p_samples_x), p_samples_x) / (p_samples_x.shape[0])
    n_var_x = np.matmul(np.transpose(n_samples_x), n_samples_x) / (n_samples_x.shape[0])
    tmp_diff = (p_avg_x - n_avg_x) 
    tmp_diff = tmp_diff[:, np.newaxis]
    tmp_diff_T = np.transpose(tmp_diff)
    s_b = np.matmul(tmp_diff, tmp_diff_T)
    s_w = np.add(p_var_x, n_var_x).astype(np.float32)
    avg_diff = p_avg_x - n_avg_x
    avg_diff = avg_diff[:, np.newaxis].astype(np.float32)
    return s_b, s_w, avg_diff

def wrong_num(labels, datas, model):
    num = 0
    pred = model(datas)
    for x, y in zip(labels, pred):
        x = x[0]
        y = y.numpy()[0]
        if x * y < 0:
            num += 1
    return num

class LDA(tf.keras.Model):
    def __init__(self):
        super(LDA, self).__init__()
        self.w  = tf.Variable(tf.random.normal([2, 1]), name="w")
        
    def call(self, s_b, s_w):
        x = tf.matmul(tf.transpose(self.w), s_b)
        x = tf.matmul(x, self.w)
        y = tf.matmul(tf.transpose(self.w), s_w)
        y = tf.matmul(y, self.w)
        return - x / y
        
@tf.function
def train_step(s_b, s_w, model):
    with tf.GradientTape() as tape:
        J = model(s_b, s_w)
    grad = tape.gradient(J, model.trainable_variables)
    opti.apply_gradients(zip(grad, model.trainable_variables))
    return J 

def draw_picture(datas, cosine_list, w, w_form):
    p_samples = [it for it in datas if plane_function(it[0], it[1]) > 0]
    p_samples_x = [x for x, y in p_samples]
    p_samples_y = [y for x, y in p_samples]

    n_samples = [it for it in datas if plane_function(it[0], it[1]) < 0]
    n_samples_x = [x for x, y in n_samples]
    n_samples_y = [y for x, y in n_samples]

    plt.scatter(p_samples_x, p_samples_y, c="red")
    plt.scatter(n_samples_x, n_samples_y, c="blue")

    w1 = w[0][0]
    w2 = w[1][0]
    t = np.linspace(-100, 100, num=10)
    x = list()
    y = list()
    for i in t:
        x.append(i * w1)
        y.append(i * w2)
    plt.plot(x, y, color="blue", marker="*")
    w1 = w_form[0][0]
    w2 = w_form[1][0]
    t = np.linspace(-1000, 1000, num=10)
    x = list()
    y = list()
    for i in t:
        x.append(i * w1)
        y.append(i * w2)
    plt.plot(x, y, color="red", marker="^")
    plt.show()
    plt.plot(list(range(len(cosine_list))), cosine_list)
    plt.show()

def cosine_sim(x, y):
    x_l2_norm = tf.linalg.norm(x)
    y_l2_norm = tf.linalg.norm(y)
    dot_xy = tf.matmul(tf.transpose(x), y)
    cosine = tf.divide(dot_xy, x_l2_norm * y_l2_norm)
    return cosine

if __name__ == "__main__":
    # 生成数据
    datas, labels = data_generator(1000)
    s_b, s_w, avg_diff = process_data(datas, labels)
    # 定义模型
    lda = LDA()
    # 定义优化器
    opti = tf.keras.optimizers.Adam()
    # 模型参数优化
    wrong_num_list = list()
    cosine_list = list()
    # 目标w
    w = tf.constant([[1],[2]], dtype=tf.float32)

    for i in range(10000):
        J = train_step(s_b, s_w, lda)
        if i % 100 == 0:
            cosine = cosine_sim(w, lda.w)
            cosine = tf.squeeze(cosine)
            cosine = tf.abs(cosine)
            cosine_list.append(cosine)
            print("epoch: {}, cosine: {}".format(i, cosine))
    w = lda.w.numpy()
    w_form = tf.matmul(tf.linalg.inv(s_w), avg_diff)
    draw_picture(datas, cosine_list, lda.w.numpy(), w_form)


