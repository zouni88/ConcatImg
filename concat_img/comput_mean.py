import math
import os

import numpy as np
import cv2
from tqdm import tqdm


# base = r'E:\face\faces1'
base = 'e:/photo1'
faces = os.listdir(base)
groups = 2

faceP = os.path.join(base, faces[0])
abc = cv2.imread(faceP)
abc
base
def getMean(img):
    mr = img[:, :, 0].mean()
    mg = img[:, :, 1].mean()
    mb = img[:, :, 2].mean()
    return mr + mg + mb


def group():
    means = {}
    for i in tqdm(faces):
        faceP = os.path.join(base, i)
        faceI = cv2.imread(faceP)
        mean = getMean(faceI)
        means[faceP] = mean
    # 对所有均值进行排序，从小到大 升序排列
    list1 = sorted(means.items(), key=lambda x: x[1])
    nparray = np.array([*means.values()])
    maxv = math.floor(nparray.max())
    minv = math.floor(nparray.min())
    step = math.floor((maxv - minv) / groups)
    steam = {}
    for i in tqdm(range(minv, maxv, step)):
        temp = []
        for item in list1:
            if item[1] > i and item[1] < i + step:
                temp.append(item)
        if len(temp) > 0:
            steam[i] = temp
    return steam


def getPathsByMean(steam, mean):
    faces = []
    for i, item in enumerate(steam.keys()):
        if mean >= item and i < len(steam.keys()) - 1 and mean < [*steam.keys()][i + 1]:
            faces = steam[item]
        if i == len(steam.keys()) - 1 and mean >= item:
            faces = steam[item]
        if mean < item and i == 0:
            faces = steam[item]
        if mean > item and i == len(steam.keys()) - 1:
            faces = steam[item]

    temp = []
    for i in faces:
        temp.append(i[0])
    faces = temp
    return faces


if __name__ == '__main__':
    steam = group()
    mean = 10000
    faces = []
    for i, item in enumerate(steam.keys()):
        if mean >= item and i < len(steam.keys()) - 1 and mean < [*steam.keys()][i + 1]:
            faces = steam[item]
        if i == len(steam.keys()) - 1 and mean >= item:
            faces = steam[item]
        if mean < item and i == 0:
            faces = steam[item]
        if mean > item and i == len(steam.keys()) - 1:
            faces = steam[item]
    print(faces)
    temp = []
    for i in faces:
        print(i[0])
        temp.append(i[0])
