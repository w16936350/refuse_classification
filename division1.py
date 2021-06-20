
import os
import random
import numpy as np
"""
把文件名及类别写入文件
"""

class_name_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}  # 给分类编号
data_dir = 'dataset/'  # 读取垃圾数据集的路径
output_path = 'list.txt'  # 文件名存放的txt文件名
fd = open(output_path, 'w')  # 文件名存放的路径
for class_name in class_name_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)  # 返回一个由文件名和目录名组成的列表
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name,
                                     class_name_to_ids[class_name]))  # 向文件写入格式为 分类名称/图片名称 分类编号 的数据
fd.close()  # 关闭文件


