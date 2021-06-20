# 随机选取样本做训练集和验证集
import random
_NUM_VALIDATION = 505  # 取505个样本作为验证集（25%）
_RANDOM_SEED = 0
list_path = 'list.txt'  # 完整分类数据的txt文件
train_list_path = 'list_train.txt'  # 训练集的数据文件
val_list_path = 'list_val.txt'  # 验证集的数据文件
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)  # 打乱文件数据
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:  # 随机分入训练集，取下标为505之后的数据样本作为训练集
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:  # 随机分入验证集，取前505个数据样本作为验证集
    fd.write(line)
fd.close()
