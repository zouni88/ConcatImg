import colorsys
import numpy as np
import cv2

from image_deal.concat_img.getdoc.PrimaryColor import exter


def same1(img1, img2):
    mr = img1[:, :, 0].mean()
    mg = img1[:, :, 1].mean()
    mb = img1[:, :, 2].mean()

    m1r = img2[:, :, 0].mean()
    m1g = img2[:, :, 1].mean()
    m1b = img2[:, :, 2].mean()

    mean = np.abs(mr - m1r) + np.abs(mg - m1g) + np.abs(mb - m1b)
    if mean < 20:
        return 0.8
    return 0.1


def same(img1, img2):
    img2 = cv2.resize(img2, dsize=(10, 10))
    i1 = exter(img1)
    i2 = exter(img2)

    r = i1[0] - i2[0]
    g = i1[1] - i2[1]
    b = i1[2] - i2[2]
    if np.abs(r) < 30 and np.abs(g) < 30 and np.abs(b) < 30:
        return 0.9
    else:
        return 0


def get_dominant_color(image):
    # 颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

    # 生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0  # 原来的代码此处为None
    dominant_color = 0  # 原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 score > max_score的比较中，max_score的初始格式不定

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color
