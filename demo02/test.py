import tensorflow as tf
import os
import numpy as np
from PIL import Image
from training import crack_captcha_cnn, vec2text
from training import X, MAX_CAPTCHA, CHAR_SET_LEN, name2vec
import random

def get_name_and_image():
    all_image = os.listdir('test_imgs/')
    random_file = random.randint(0, 200)
    name =  all_image[random_file]
    image = Image.open('test_imgs/'+ name)
    image = np.expand_dims(np.array(image), axis=2)
    image = np.expand_dims(np.array(image), axis=0)
    return name[:-4], image

output = crack_captcha_cnn()
saver = tf.train.Saver()
with tf.Session() as sess:
    n = 0
    while n<200:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        text, image = get_name_and_image()
        predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)

        text_list = sess.run(max_idx_p, feed_dict={X: image})
        vec = text_list[0].tolist()
        # for t in predict_text:
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        for idx, num in enumerate(vec):
            vector[idx * 36 + num] = 1
        predict_text = vec2text(vector)
        print(f"正确：{text} 预测: {predict_text}")
        n +=1


