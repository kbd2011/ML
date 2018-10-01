__author__ = 'Krunal'

#from __future__ import print_function
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import h5py as db
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from numpy import genfromtxt
import os

def load_testdata(path):
    x_test = []
    print 'test'
    validlist = []
    listing = os.listdir(path+'test\\')
    if os.path.isfile(path + 'xtest.npy'):
        x_test= np.load(path + 'xtest.npy')
        imgname=np.load(path + 'imgname.npy')
    else:
        for r in listing:
            img1 = cv2.imread(path+'test\\'+r)
            if img1 is not None:
                img = cv2.resize(img1,(28,28))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                x_test.append(img)
            else:
                img = np.zeros((28, 28), dtype=float)
                x_test.append(img)
        x_test = np.array(x_test)
        imgname = map(lambda imgname: imgname.replace('.jpg', ''), listing)
        imgname = np.array(imgname)
        imgname = imgname.astype(np.int)
        np.save(path + 'xtest.npy', x_test)
        np.save(path + 'imgname.npy', imgname)

    print 'test data set shape'
    print x_test.shape
    print imgname.shape
    return imgname,x_test


##########  31 Layers Network
def VGG_111Grey(path):
    print 'VGG111Grey'
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 28, 28)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    if os.path.isfile(path + 'VGG1_weights_grey111.h5'):
        print 'Weight used'
        model.load_weights(path + 'VGG1_weights_grey111.h5')
    return model


def main(args):
    path = 'C:\\Users\\Krunal\\Documents\\DSBA\\Spring 2016\\ML\\Project\\project\\'
    img_name,X_test=load_testdata(path)

    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_test = X_test.astype('float32')
    X_test /= 255
    print(X_test.shape[0], 'test samples')
        # convert class vectors to binary class matrices

    model = VGG_111Grey(path)
    output = model.predict_proba(X_test,batch_size=32,verbose=1)
    y_test=np.column_stack((img_name,output)).astype(np.float)
    np.savetxt(path+'\\test.csv',y_test,header="id,col1,col2,col3,col4,col5,col6,col7,col8",comments='',delimiter=',',fmt='%i,%f,%f,%f,%f,%f,%f,%f,%f')

if __name__ == '__main__':
    main(sys.argv)

