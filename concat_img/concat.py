import os
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import math
from tqdm import tqdm

from concat_img.comput_mean import group, getMean, getPathsByMean

# 初始化参数
param_px = 30

rootPath = 'E:/face'
# base = r'E:\face\faces1'
base = 'e:/photo1'
opath = rootPath + '/child.jpg'
# 将原图放大十倍
org_img = cv2.imread(opath)
org_img = cv2.resize(org_img, dsize=(org_img.shape[1] * 10, org_img.shape[0] * 10), interpolation=cv2.INTER_NEAREST)

w = org_img.shape[0]
h = org_img.shape[1]

usedFaces = []
# 初始化 对所有数据进行分组 缩短遍历时长
steam = group()


def same1(img1, img2):
    def getMean(img):
        mr = img[:, :, 0].mean()
        mg = img[:, :, 1].mean()
        mb = img[:, :, 2].mean()
        return mr, mg, mb

    mr, mg, mb = getMean(img1)
    m1r, m1g, m1b = getMean(img2)
    return np.abs(mr - m1r) + np.abs(mg - m1g) + np.abs(mb - m1b)


def compuSim(img):
    mean = getMean(img)
    faces = getPathsByMean(steam, mean)
    temp = []
    for i, face in enumerate(faces):
        faceP = os.path.join(base, face)
        faceI = cv2.imread(faceP)
        # 计算图片色值相似度
        degree = same1(img, faceI)
        temp.append(degree)
    temp = np.array(temp)
    index = np.argmin(temp)
    faceP = os.path.join(base, faces[index])
    faceI = cv2.imread(faceP)

    h = faceI.shape[0]
    w = faceI.shape[1]
    if h > w:
        faceI = faceI[:w, :, :]
    else:
        faceI = faceI[:, :h, :]

    faceI = cv2.resize(faceI, dsize=(param_px, param_px), interpolation=cv2.INTER_NEAREST)
    return faceI


def item(oriImg, startx, endx, starty, endy):
    temp = np.zeros([param_px, param_px, 3], dtype='uint8')
    for c in range(3):
        for i, x in enumerate(range(startx, endx)):
            for j, y in enumerate(range(starty, endy)):
                temp[i, j, c] = oriImg[x, y, c]
    # 获取相似度图片
    insImg = compuSim(temp)
    for c in range(3):
        for i, x in enumerate(range(startx, endx)):
            for j, y in enumerate(range(starty, endy)):
                oriImg[x, y, c] = insImg[i, j, c]

    return oriImg


# 需要轮询的次数 图片宽高/10 向上取整
iterNum_row = math.floor(org_img.shape[0] / param_px)
iterNum_col = math.floor(org_img.shape[1] / param_px)
# 初始轮询坐标点
startx, endx, starty, endy = 0, param_px - 1, 0, param_px - 1
for r in tqdm(range(iterNum_row)):
    print('================第 %d 行' % r)
    for c in tqdm(range(iterNum_col)):
        oriImg = item(oriImg=org_img, startx=startx, endx=endx, starty=starty, endy=endy)
        org_img = oriImg
        starty += param_px if starty != 0 else param_px - 1
        endy += param_px
    starty = 0
    endy = param_px - 1
    startx += param_px if startx != 0 else param_px - 1
    endx += param_px
    # if r == 0:
    #     break
    cv2.imwrite(rootPath + '/123.jpg', org_img)


def test():
    faces = os.listdir(base)
    faceP = os.path.join(base, faces[0])
    faceI = cv2.imread(faceP)

    h = faceI.shape[0]
    w = faceI.shape[1]
    if h > w:
        faceI = faceI[:w, :, :]
    else:
        faceI = faceI[:, :h, :]

    faceI[:150, :, :].shape
    faceI.shape
