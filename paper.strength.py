import PIL
from PIL import Image,ImageEnhance,ImageChops
import numpy as np
import random
import os

#打开图片
def open_img(image):
    return Image.open(image,mode="r")

#1、旋转变换rotation
def rotation(image,mode = Image.BICUBIC):
    random_angle = np.random.randint(1,360)  #随机旋转1~360°
    return image.rotate(random_angle,mode)

#2、平移变换shift
def shift(image,off_x = 0,off_y = 0):
    return ImageChops.offset(image,off_x,off_y)

#3、随机修剪crop
def crop(image):
    image_width = image.size[0]
    image_height = image.size[1]  #取出原始图片的大小
    crop_win_size = np.random.randint(min(image_width,image_height) * 0.6,min(image_width,image_height))
                                                                                   #随机产生修剪框的大小
    random_region = (
        (image_width-crop_win_size) >>1,(image_height-crop_win_size) >>1,
        (image_width+crop_win_size) >>1,(image_height+crop_win_size) >>1
    )
    return image.crop(random_region)


#4、随机翻转flip
def flip(image,mode = 0):
    if mode == 0:
        return image.transpose(Image.FLIP_LEFT_RIGHT)  #左右翻转
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)  #上下翻转


#5、对比度增强contrast
def contrast_enhancement(image):
    enh_con = ImageEnhance.Contrast(image)
    contrast =  1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


#6、亮度增强brightness
def brightness_enhancement(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brighted = enh_bri.enhance(brightness)
    return image_brighted


#7、色彩抖动
def random_color(image):
    random_factor = np.random.randint(0,31) / 10
    color_image = ImageEnhance.Color(image).enhance(random_factor)  #调整饱和度
    random_factor = np.random.randint(10, 21) / 10
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  #调整亮度
    random_factor = np.random.randint(10, 21) / 10
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  #调整对比度
    random_factor = np.random.randint(0, 31) / 10
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  #调整锐度


#8、噪声扰动
def gaussian(image,mean = 0.2,sigma = 0.3):
    def gaussian_noisy(im,mean = 0.2,sigma = 0.3):
        """
        对图像进行高斯噪音处理
        :param im:单通道图像
        :param mean:偏移量
        :param sigma:标准差
        :return:
        """
        for i in range(len(im)):
            im[i] += random.gauss(mean,sigma)
        return im

    img = np.array(image)  #将图像转化成数组
    img.flags.writeable = True
    width , height = img.shape[:2]  #将数组改成读写模式
    img_r = gaussian_noisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussian_noisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussian_noisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


for path,obj,lists in os.walk("paper_img"):
    for i in lists:
        # 随机旋转
        rotate = rotation(open_img('paper_img/'+i))
        rotate.save('paper_save/' + i[:-4] + '_' + str(1) + '.png')
        # 随机平移
        trans = shift(open_img('paper_img/'+i), random.randint(0, 5000), random.randint(0, 5000))
        trans.save('paper_save/' + i[:-4] + '_' + str(2) + '.png')
        # 随机修剪
        crp = crop(open_img('paper_img/'+i))
        crp.save('paper_save/' + i[:-4] + '_' + str(3) + '.png')
        # 左右翻转
        flip_h = flip(open_img('paper_img/'+i), 0)
        flip_h.save('paper_save/' + i[:-4] + '_' + str(4) + '.png')
        # 上下翻转
        flip_v = flip(open_img('paper_img/'+i), 1)
        flip_v.save('paper_save/' + i[:-4] + '_' + str(5) + '.png')
        # 对比度增强
        contrast = contrast_enhancement(open_img('paper_img/'+i))
        contrast.save('paper_save/' + i[:-4] + '_' + str(6) + '.png')
        # 亮度增强
        bright = brightness_enhancement(open_img('paper_img/'+i))
        bright.save('paper_save/' + i[:-4] + '_' + str(7) + '.png')
        # 色彩抖动
        color = random_color(open_img('paper_img/'+i))
        color.save('paper_save/' + i[:-4] + '_' + str(8) + '.png')
        # 噪声扰动
        noise = gaussian(open_img('paper_img/'+i))
        noise.save('paper_save/' + i[:-4] + '_' + str(9) + '.png')