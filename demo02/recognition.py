from PIL import Image, ImageFilter, ImageChops, ImageColor, ImageDraw
import pytesseract
import requests
import os
from queue import Queue
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
            print(f'download {i} image')
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


def cfs(im, x_fd, y_fd):
    '''用队列和集合记录遍历过的像素坐标代替单纯递归以解决cfs访问过深问题'''
    xaxis = []
    yaxis = []
    visited = set()
    q = Queue()
    q.put((x_fd, y_fd))
    visited.add((x_fd, y_fd))
    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 四邻域

    while not q.empty():
        x, y = q.get()

        for x_offset, y_offset in offsets:
            x_neighbor, y_neighbor = x + x_offset, y + y_offset

            if (x_neighbor, y_neighbor) in visited:
                continue

            visited.add((x_neighbor, y_neighbor))

            try:
                if im.getpixel((x_neighbor, y_neighbor)) == 0:
                    q.put((x_neighbor, y_neighbor))
                    xaxis.append(x_neighbor)
                    yaxis.append(y_neighbor)
            except IndexError:
                pass
    xmax = max(xaxis)
    xmin = min(xaxis)
    ymax = max(yaxis)
    ymin = min(yaxis)
    return xmin, xmax, ymin, ymax


def detectFgPix(im, xmax):
    '''搜索区块起点
    '''

    w, h = im.size
    for x_fd in range(xmax + 1, w):
        for y_fd in range(h):
            if im.getpixel((x_fd, y_fd)) == 0:
                return x_fd, y_fd


def CFS(im):
    '''切割字符位置
    '''

    zoneL = []  # 各区块长度L列表
    zoneWB = []  # 各区块的X轴[起始，终点]列表
    zoneHB = []  # 各区块的Y轴[起始，终点]列表

    xmax = 0  # 上一区块结束黑点横坐标,这里是初始化
    for i in range(10):
        try:
            x_fd, y_fd = detectFgPix(im, xmax)
            # print(x_fd, y_fd)
            xmin, xmax, ymin, ymax = cfs(im, x_fd, y_fd)
            L = xmax - xmin
            H = ymax - ymin
            zoneL.append(L)
            zoneWB.append([xmin, xmax])
            zoneHB.append([ymin, ymax])

        except Exception as e:
            return zoneL, zoneWB, zoneHB

    return zoneL, zoneWB, zoneHB


def cutting_img(im, im_position, img, xoffset=1, yoffset=1):
    if not os.path.exists('./out_img'):
        os.mkdir('./out_img')
    filename = './out_img/' + img.split('.')[0]
    print(filename)
    # 识别出的字符个数
    im_number = len(im_position[1])
    # 切割字符
    for i in range(im_number):
        # left, upper, right, and lower
        left = im_position[1][i][0] - xoffset
        right = im_position[1][i][1] + xoffset
        upper = im_position[2][i][0] - yoffset
        lower = im_position[2][i][1] + yoffset
        cropped = im.crop((left, upper, right, lower))
        cropped.save(filename + '-cutting-' + str(i) + '.jpg')


def main():
    filedir = './imgs'
    for imgname in os.listdir(filedir):
        if fnmatch(imgname, '*.png'):
            img = binaryzation(filedir, imgname)
            img = depoint(img)  # clear point
            img = clear_border(img, 3)
            # img.show()
            print(imgname, pytesseract.image_to_string(img, lang='eng'))

            # # 切割的位置
            # im_position = CFS(img)
            # maxL = max(im_position[0])
            # minL = min(im_position[0])
            #
            # # 如果有粘连字符，如果一个字符的长度过长就认为是粘连字符，并从中间进行切割
            # if (maxL > minL + minL * 1.5):
            #     maxL_index = im_position[0].index(maxL)
            #     minL_index = im_position[0].index(minL)
            #     # 设置字符的宽度
            #     im_position[0][maxL_index] = maxL // 2
            #     im_position[0].insert(maxL_index + 1, maxL // 2)
            #     # 设置字符X轴[起始，终点]位置
            #     im_position[1][maxL_index][1] = im_position[1][maxL_index][0] + maxL // 2
            #     im_position[1].insert(maxL_index + 1, [im_position[1][maxL_index][1] + 1,
            #                                            im_position[1][maxL_index][1] + 1 + maxL // 2])
            #     # 设置字符的Y轴[起始，终点]位置
            #     im_position[2].insert(maxL_index + 1, im_position[2][maxL_index])
            # # 切割字符，要想切得好就得配置参数，通常 1 or 2 就可以
            # cutting_img(img, im_position, imgname, 1, 1)


if __name__ == '__main__':
    # get_images()
    main()

