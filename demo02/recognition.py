from PIL import Image, ImageFilter, ImageChops, ImageColor, ImageDraw
import pytesseract
import requests
import os
from fnmatch import fnmatch


def get_images():
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    url = 'http://jwxt.wust.edu.cn//whkjdx/verifycode.servlet?0.15510035031440483'
    try:
        for i in range(10):
            r = requests.get(url)
            r.raise_for_status()
            filename = f'imgs/{str(i)}.png'
            with open(filename, 'wb') as fp:
                fp.write(r.content)
    except Exception as e:
        print('status:', r.status_code)


def binaryzation(filedir, imgname):
    img = Image.open(filedir + '/' + imgname).convert('L')
    img2 = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(img2)
    for i in range(0, img.size[1]):
        for j in range(0, img.size[0]):
            pix = img.getpixel((j, i))
            if pix < 130:
                draw.point((j, i), fill=0)
    return img2


def clear_border(img, width):
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


def main():
    filedir = './imgs'
    for imgname in os.listdir(filedir):
        if fnmatch(imgname, '*.png'):
            img = binaryzation(filedir, imgname)
            img = depoint(img)
            img = clear_border(img, 1)
            img.show()
            print(imgname, pytesseract.image_to_string(img))


if __name__ == '__main__':
    main()
