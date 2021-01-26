# get the target image
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import copy

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n' 
        file.write(s)
    file.close()
    print("保存文件成功")

def get_area1(path,gap = 100):
    all_img = cv2.imread(str(path))
    ORIG_SHAPE = all_img.shape
    orig_y,orig_x=ORIG_SHAPE[0],ORIG_SHAPE[1]
    ORIG_AREA = all_img.shape[0]*all_img.shape[1]
    
    gray_img = cv2.imread(str(path),0)
    thresh_original = cv2.adaptiveThreshold(src=gray_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            thresholdType=cv2.THRESH_BINARY, blockSize=10001,C=-20)

    # Now find contours in it.
    thresh = copy.copy(thresh_original)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # get contours with highest height
    lst_contours = []
    for cnt in contours:
        ctr = cv2.boundingRect(cnt)
        lst_contours.append(ctr)
    x,y,w,h = sorted(lst_contours, key=lambda coef: coef[3])[-1]

    # draw contours
    all_img = all_img[max(y-gap,0):min(y+h+gap,orig_y), max(x-gap,0):min(x+w+gap,orig_x)]

    CROP_SHAPE = all_img.shape
    CROP_AREA = (w+2*gap) * (h+2*gap)
    return all_img, max(x-gap,0), max(y-gap,0)


pathimgs = Path('/home/aistudio/PaddleDetection/dataset/coco/tile_round1_testA_20201231/testA_imgs') #厕所文件夹
train_imgs = os.listdir(pathimgs)
train_imgs.sort()

print("{} 内数据数量 {}".format(pathimgs,len(train_imgs)))
print('='*50)

pathimgs_crop = str(pathimgs)+"_crop"
if not os.path.exists(pathimgs_crop):
    os.mkdir(pathimgs_crop) 

data = []
for index in tqdm(range(len(train_imgs))):
    img_name = train_imgs[index]
    now_img_path = pathimgs / img_name
    img, xmin, ymin = get_area1(now_img_path)    
    filename = pathimgs_crop + '/' + img_name
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    data.append([img_name,xmin, ymin])
text_save(pathimgs_crop+'_data.txt', data) # txt里面保存图片名和截黑边的开始坐标xy