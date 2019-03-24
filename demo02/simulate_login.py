import requests
import tensorflow as tf
import io
from PIL import Image
import numpy as np
from training import crack_captcha_cnn, X, vec2text
from get_image import binaryzation, clear_border, depoint
from bs4 import BeautifulSoup

# 这里我演示的就是本人所在学校的教务系统
#

class LoginJust():
    def __init__(self, post_url, text_url, img_url, header, account, pwd):
        self.post_url = post_url
        self.text_url = text_url
        self.img_url = img_url
        self.header = header
        self.account = account
        self.pwd = pwd
        self.sess = requests.session()
        self.sess.headers = self.header

    # 获取验证码图片
    def getImage(self):
        self.sess.get(self.post_url)  # 进入登陆页面
        r = self.sess.get(self.img_url)
        byte_stream = io.BytesIO(r.content)  # 把请求到的数据转换为Bytes字节流
        img = Image.open(byte_stream)  # Image打开二进制流Byte字节流数据
        img = binaryzation(img)
        img = clear_border(img)
        img = depoint(img)
        img.show()
        img = np.expand_dims(np.array(img), axis=2)
        img = np.expand_dims(img, axis=0)
        return img

    # 识别验证码
    def recognize(self):
        image = self.getImage()
        output = crack_captcha_cnn()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            predict = tf.reshape(output, [-1, 4, 36])
            max_idx_p = tf.argmax(predict, 2)
            text_list = sess.run(max_idx_p, feed_dict={X: image})
            vec = text_list[0].tolist()
            vector = np.zeros(4 * 36)
            for idx, num in enumerate(vec):
                vector[idx * 36 + num] = 1
            predict_text = vec2text(vector)
        return predict_text

    # 登陆login
    def login(self):
        captcha_text = self.recognize()
        data = {
            'USERNAME': self.account,
            'PASSWORD': self.pwd,
            'useDogCode': '',
            'useDogCode': '',
            'RANDOMCODE': captcha_text.lower(),
            'x': '75',
            'y': '20'
        }
        r = self.sess.post(self.post_url, data=data)

    # 获取登陆后的内容
    def get_text(self):
        self.login()
        r = self.sess.get(text_url)
        self.IsLoginS(r.text)


    # 判断登陆是否成功
    def IsLoginS(self, html):
        soup = BeautifulSoup(html, 'lxml')
        title = soup.find('title').text
        if self.account in title:
            print('登陆成功')
        else:
            print('登陆失败')


# 需要post的网址的URL
post_url = 'http://jwxt.wust.edu.cn/whkjdx/Logon.do?method=logon'

# 获取后台数据的网址
text_url = 'http://jwxt.wust.edu.cn/whkjdx/framework/main.jsp'

# 获取验证码图片的地址
img_url = 'http://jwxt.wust.edu.cn/whkjdx/verifycode.servlet'

# header
header = {
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Host': 'jwxt.wust.edu.cn',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
}
account = input("请输入用户名：")
pwd = input("请输入密码：")
# 实例化对象
lj = LoginJust(post_url, text_url, img_url, header, account, pwd)

lj.get_text()

