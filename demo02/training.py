from PIL import Image, ImageFilter, ImageChops, ImageColor, ImageDraw
import requests
import tensorflow as tf
import numpy as np
import os
import math
from queue import Queue
from fnmatch import fnmatch


IMAGE_HEIGHT = 22
IMAGE_WIDTH = 62
MAX_CAPTCHA = 4
CHAR_SET_LEN = 36  # 26 + 10
X = tf.placeholder(tf.float32, [None, 22, 62, 1])
Y = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * MAX_CAPTCHA])
keep_prob = tf.placeholder(tf.float32)  # dropout


def name2vec(name):

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 32 - 65 + 10
                # k = ord(c) - 61
                if k > 35:
                    raise ValueError('No Map')
        return k

    for i , c in enumerate(name[:4]):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        # elif char_idx < 62:
        #     char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

def get_image_and_label(batch_img_path, dir='imgs/'):
    batch_x = np.zeros([batch_img_path.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    batch_y = np.zeros([batch_img_path.shape[0], MAX_CAPTCHA * CHAR_SET_LEN])
    for index, img_name in enumerate(batch_img_path):
        img = Image.open(dir + img_name)
        batch_x[index, :] = np.expand_dims(np.array(img), axis=2)
        batch_y[index, :] = name2vec(img_name)
    return batch_x, batch_y

def random_mini_batches(mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from img_path

    Arguments:
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of imgpath
    """
    img_paths = np.array(os.listdir('imgs/'))
    m = img_paths.shape[0]   # number of training examples
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    mini_batches = []

    permutation = list(np.random.permutation(m))

    img_paths = img_paths[permutation]

    # Step 2: Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_path = img_paths[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batches.append(mini_batch_path)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_path = img_paths[num_complete_minibatches * mini_batch_size:]
        mini_batches.append(mini_batch_path)

    return mini_batches

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01):
    n = X.shape[0]  # example num
    # CONV2D: stride of 1, padding 'SAME'
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    Z1 = tf.nn.conv2d(input=X, filter=w_c1, strides=(1, 1, 1, 1), padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(value=A1, ksize=(1, 3, 3, 1), strides=(1, 3, 3, 1), padding='SAME')
    # assert (P1.shape == (n, 8, 21, 32))

    # CONV2D: filters W2, stride 1, padding 'SAME'
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    Z2 = tf.nn.conv2d(input=P1, filter=w_c2, strides=(1, 1, 1, 1), padding='SAME')
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(value=A2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    # assert (P2.shape == (n, 4, 11, 64))

    # FLATTEN
    # P2 = tf.contrib.layers.flatten(inputs=P2)  # flatten is deperacated and will be romoved
    P2 = tf.reshape(P2, (-1, P2.shape[1] * P2.shape[2] * P2.shape[3]))
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, MAX_CAPTCHA * CHAR_SET_LEN, activation_fn=None)
    # print(Z3.shape)
    return Z3


# 训练
def train_crack_captcha_cnn():
    import time
    start_time = time.time()
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    # Calculate the correct predictions
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize all the variables globally
        sess.run(tf.global_variables_initializer())

        epoch = 0
        while 1:
            batch_img_paths = random_mini_batches(mini_batch_size=64, seed=epoch)  # img path batch
            for batch_img_path in batch_img_paths:
                batch_x, batch_y = get_image_and_label(batch_img_path)
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'--', epoch, '--', loss_)

            # 每100 step计算一次准确率
            if epoch % 10 == 0:
                batch_x, batch_y = get_image_and_label(batch_img_path)
                acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
                print(u'*****************第%s次的准确率为%s' % (epoch, acc))
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.9:  ##我这里设了0.9，设得越大训练要花的时间越长，如果设得过于接近1，很难达到。如果使用cpu，花的时间很长，cpu占用很高电脑发烫。
                    saver.save(sess, "crack_capcha.model", global_step=epoch)
                    print(time.time() - start_time)
                    break
            epoch += 1


if __name__ == '__main__':
    train_crack_captcha_cnn()
    # output = crack_captcha_cnn()
    # predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    # max_idx_p = tf.argmax(predict, 2)
    # max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    #
    # correct_pred = tf.equal(max_idx_p, max_idx_l)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for i in os.listdir('imgs'):
    #         batch_x, batch_y = get_image_and_label(np.array(os.listdir('imgs')[:5]))
    #         # print(i, vec2text(batch_y[0,:]))
    #         print(os.listdir('imgs')[:5], vec2text(batch_y))
    #         # text_list = sess.run(max_idx_p, feed_dict={X: batch_x})
    #         # print(text_list)
    #         text_list = sess.run(max_idx_l, feed_dict={Y :batch_y})
    #         # print(text_list)
    #         # vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    #         # i = 0
    #         for t in text_list:
    #             vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    #             for idx, num in enumerate(t):
    #                 vector[idx * 36 + num] = 1
    #             print(vec2text(vector))
    #         acc = sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y})
    #         # print(f"正确：{i} 预测: {predict_text}")
    #         break

