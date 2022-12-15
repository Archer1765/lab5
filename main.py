from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train / 255
x_test = x_test / 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(800, input_dim = 784, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

model.compile(loss="categorical_crossentropy", optimizer = "SGD", metrics = ["accuracy"])

#print(model.summary())

history = model.fit(x_train, y_train, batch_size=200, epochs=50,validation_split=0.2,verbose=1)

scores = model.evaluate(x_test, y_test, verbose=1)
print("Процент распознования тестовых данных: ", round(scores[1]*100, 4))

model.save('my_model.h5')


