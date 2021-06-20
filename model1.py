from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical
import os
import numpy as np


from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_train_test_data(list_file):  # 写一个函数用于处理数据的函数
    list_train = open(list_file)
    x_train = []
    y_train = []
    for line in list_train.readlines():
        x_train.append(line.strip()[:-2])  # 剔除空格换行，取其图片的路径名称
        y_train.append(int(line.strip()[-1]))  # 提出空格换行，取其图片的分类编号（标签）
    return x_train, y_train


x_train, y_train = get_train_test_data('list_train.txt')
x_test, y_test = get_train_test_data('list_val.txt')


def process_train_test_data(x_path):
    images = []
    for image_path in x_path:
        img_load = load_img('dataset/' + image_path)
        img = image.img_to_array(img_load)  # 将图片的数据转化成数组
        img = preprocess_input(img)  # 对图片数组进行预处理，转换图片格式适配模型，能够加快图像的处理速度
        images.append(img)
    return images


train_images = process_train_test_data(x_train)  # 获得训练集图片
test_images = process_train_test_data(x_test)  # 获得测试集图片

# 构造模型（keras中的Inception_resnet_V2）
base_model = InceptionResNetV2(include_top=False, pooling='avg')  # 不加载最后一层的权重，同时进行average池化
outputs = Dense(6, activation='softmax')(base_model.output)  # 构造六分类，用‘softmax’激活，把模型前面的几层的输出作为最后一层的输入
model = Model(base_model.inputs, outputs)  # 构造模型

# 设置ModelCheckpoint，按照验证集的准确率进行保存
save_dir = 'train_model'
filepath = "model_{epoch:02d}-{val_accuracy:.2f}.hdf5"  # 保存路径，保存了模型训练了几代，模型成功率等信息
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath),
                             monitor='val_accuracy', verbose=1,  # 以模型准确率作为监视，把结果设置为可视化
                             save_best_only=True)  # 只保存最好的结果


# 模型设置
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)  # 预测的概率为前三位的准确率


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)  # 预测的概率为前三位的准确率


model.compile(optimizer='adam',  # 优化器为adam
              loss='categorical_crossentropy',  # 损失函数为交叉损失
              metrics=['accuracy', acc_top3, acc_top5])  # 监控准确率、前三准确率、前五准确率


# 模型训练
model.fit(np.array(train_images), to_categorical(y_train),
          batch_size=8,  # 一次训练所选取的样本数为8
          epochs=5,  # 训练5代
          shuffle=True,  # 训练时打乱
          validation_data=(np.array(test_images), to_categorical(y_test)),  # 验证集
          callbacks=[checkpoint])  # 用准确率的checkpoint作为回调函数参数


# 加载指定模型
model.load_weights('train_model/model_05-0.80.hdf5')

# 直接使用predict方法进行预测
y_pred = model.predict(np.array(test_images))
print(y_pred)
ans = []
count = 0
for l in y_pred:
    ac = 0
    for i in range(len(l)):
        # print(ll)
        # print(type(ll))
        if l[i] > ac:
            ac = i
    ans.append(ac)

for i in range(len(y_test)):
    if y_test[i] == ans[i]:
        count += 1

print(len(y_test))
print(len(y_pred))
print(count/len(y_pred))
