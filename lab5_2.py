from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


model = models.load_model('my_model.h5')

for filename in os.listdir('image/'):
    f = os.path.join('image/', filename)
    if os.path.isfile(f) and filename.endswith('.png'):
        print('Файл: ',f)
        img = image.load_img(f, target_size=(28, 28), color_mode="grayscale")
        plt.imshow(img.convert('RGBA'))
        # plt.show()
        x = image.img_to_array(img)  # картинка в массив
        x = x.reshape(1, 784)  # форма массива в плоский вектор
        x = 255 - x  # инвертация картинки
        x /= 255  # нормализация картинки
        p = model.predict(x)
        p = np.argmax(p)
        print("Полученная цифра: ", p)
        plt.show()
'''
img_path = '999.png'
img = image.load_img(img_path, target_size=(28,28), color_mode="grayscale")
plt.imshow(img.convert('RGBA'))
#plt.show()
x = image.img_to_array(img)#картинка в массив
x = x.reshape(1, 784)#форма массива в плоский вектор
x = 255 - x#инвертация картинки
x /= 255#нормализация картинки
prediction = model.predict(x)
prediction = np.argmax(prediction)
print("Полученная цифра: ", prediction)
#print("Цифра массива: ", numbers[prediction])
plt.show()'''

