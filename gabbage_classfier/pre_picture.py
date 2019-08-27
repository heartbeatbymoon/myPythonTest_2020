# encoding=utf-8
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_dir = 'G:/datas/gabbage/test/test/'
log_dir = 'G:/datas/gabbage'
lists = ['cup', 'chazuo']


# 从测试集中随机挑选一张图片看测试结果
def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    idn = np.random.randint(0, img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    print("-------------")
    print(image_dir)
    print("-------------")
    image = Image.open(image_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([28, 28])
    image_arr = np.array(image)
    return image_arr

get_one_image(img_dir)