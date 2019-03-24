import os
import requests
import io
from PIL import Image, ImageDraw


def get_images():
    if not os.path.exists('test_imgs'):
        os.mkdir('test_imgs')
    url = 'http://jwxt.wust.edu.cn//whkjdx/verifycode.servlet?0.15510035031440483'
    try:
        for i in range(200):
            r = requests.get(url)
            r.raise_for_status()
            filename = f'test_imgs/{str(i)}.png'
            print(f'download {i} image')
            byte_stream = io.BytesIO(r.content)  # 把请求到的数据转换为Bytes字节流
            img = Image.open(byte_stream) #  Image打开二进制流Byte字节流数据
            img = binaryzation(img)
            img = clear_border(img)
            img = depoint(img)
            img.save(filename)
    except ConnectionError as e:
        print(e)
        print('status:', r.status_code)


def binaryzation(img):
    img = img.convert('L')
    img2 = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(img2)
    for i in range(0, img.size[1]):
        for j in range(0, img.size[0]):
            pix = img.getpixel((j, i))
            if pix < 130:
                draw.point((j, i), fill=0)
    return img2


def clear_border(img, width=1):
    """
    :param img: gray img
    :param width: border width
    :return:
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i in range(w):
        # up
        for j in range(width):
            draw.point((i, j), fill=255)
        # down
        for j in range(h - width, h):
            draw.point((i, j), fill=255)
    for i in range(width, h - width):
        # left
        for j in range(width):
            draw.point((j, i), fill=255)
        # right
        for j in range(w - width, w):
            draw.point((j, i), fill=255)
    return img


def depoint(img):
    # gray image
    draw = ImageDraw.Draw(img)
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            count = 0
            if img.getpixel((i, j)) == 0:
                if img.getpixel((i - 1, j)) == 255:
                    count += 1
                if img.getpixel((i + 1, j)) == 255:
                    count += 1
                if img.getpixel((i, j + 1)) == 255:
                    count += 1
                if img.getpixel((i, j - 1)) == 255:
                    count += 1
            if count > 2:
                draw.point((i, j), fill=255)
    return img


def cutting_img(im, imgname, xoffset=1, yoffset=1):
    if not os.path.exists('./traning_img'):
        os.mkdir('./traning_img')
    box0 = (2, 3, 12, 16)
    box1 = (12, 3, 22, 16)
    box2 = (22, 3, 32, 16)
    box3 = (32, 3, 42, 16)
    filename = './traning_img/'
    print(filename)
    # 切割字符
    for i in range(4):
        # left, upper, right, and lower
        cropped = im.crop(eval(f'box{i}'))
        cropped.save(imgname[i] + '/' + imgname[i] + '.jpg')



if __name__ == '__main__':
    get_images()
    # path = 'test_imgs/'
    # for i in os.listdir(path):
    #     img = Image.open(path + i)  # Image打开二进制流Byte字节流数据
    #     img = binaryzation(img)
    #     img = clear_border(img)
    #     img = depoint(img)
    #     img.save(path + i)