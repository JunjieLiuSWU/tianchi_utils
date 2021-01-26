import numpy as np
import json

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

data = read_pairs("/home/junjieliu/下载/testA_imgs_crop_data.txt") #通过get_small_test_images.py得到的txt文件
name_lists = [line[0] for line in data]

fp = open('/home/junjieliu/下载/final_results.json') #测试得到的json文件
json_data = json.load(fp)

for i in range(len(json_data)):
    # print('json_data[i]: ', json_data[i])
    name_index = name_lists.index(json_data[i]['name'])
    # print('data[name_index]: ', data[name_index])
    x = int(data[name_index][1])
    y = int(data[name_index][2])
    json_data[i]['bbox'][0] += x
    json_data[i]['bbox'][1] += y
    json_data[i]['bbox'][2] += x
    json_data[i]['bbox'][3] += y

with open('/home/junjieliu/my_result.json', 'w') as filep: # 要保存的json路径, 保存的json提交到天池
    json.dump(json_data, filep, indent=4, ensure_ascii=False)