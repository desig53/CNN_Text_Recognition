import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import PIL.ImageOps    

# Display
def plot_img(n):
    plt.imshow(X_test[n], cmap='gray')
    plt.show()


def all_img_predict(model):
    print(model.summary())
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Loss:', loss)
    print('Accuracy:', accuracy)
    predict = model.predict_classes(x_test)
    print(pd.crosstab(Y_test.reshape(-1), predict, rownames=['Label'], colnames=['predict']))


def predict_img(img,model):
    #predict = model.predict(img)
    predict = model.predict_classes(img)
    max_ = 0
    max_pos = 0
    
##    for i in range(len(predict[0])):
##        if max_<=predict[0][i]:
##            max_ = predict[0][i]
##            max_pos=i
##        print(predict[0][i])
##        print()
    
    print('Prediction:', predict[0])
    
    #print('Answer:', Y_test[n])
    #plot_img(n)

    
def load_image(filename):
    # load the image with target size
    img = load_img(filename, color_mode="grayscale",interpolation='nearest',target_size=(28,28))
    img = PIL.ImageOps.invert(img)
    img = np.reshape(img, (28, 28))
    plt.matshow(img, cmap = plt.get_cmap('gray'))  #把第一張圖畫出來
    #plt.show()

    
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28,28,1) 

    
    img = img.astype('float32')
    img = img / 255.0
    return img

# Mnist Dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train = X_train.reshape(60000, 28,28,1)/255   ##60000:多少筆資料 28,28 : vector  1:rgb
x_test = X_test.reshape(10000, 28,28,1)/255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

##print(x_train.shape)
##
### Model Structure
##model = Sequential()
##model.add(Conv2D(filters=32, kernel_size=3, input_shape=(28,28,1), activation='relu', padding='same'))
##model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
##model.add(Flatten())
##model.add(Dense(256, activation='relu'))
##model.add(Dense(10, activation='softmax'))
##
##
### Train
##model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
##model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1)
##
### Test
##loss, accuracy = model.evaluate(x_test, y_test)
##print('Test:')
##print('Loss: %s\nAccuracy: %s' % (loss, accuracy))
##
### Save model
##model.save('model.h5')

# Load Model
model = load_model('model.h5')



img = load_image("images.jpg")
predict_img(img,model)




