import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import h5py as db
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from numpy import genfromtxt
import os

def load_data(path,traincsv):
    xdata = []
    ydata = []
    x_train = []
    x_validation = []
    y_train = []
    y_validation = []

    for r in traincsv:
        img1 = cv2.imread(path+'train\\'+str(r[0])+'.jpg')
        if img1 is not None:
            img = cv2.resize(img1,(28,28))
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #### If you requires Grey images
            label = r[1:]
            xdata.append(img)
            ydata.append(label)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    np.save(path + 'vggxdatacolor28.npy', xdata)
    np.save(path + 'vggydatacolor28.npy', ydata)
    print 'train data set shape'
    print xdata.shape
    print 'label data set shape'
    print ydata.shape
    x_train, x_validation, y_train, y_validation = train_test_split(xdata, ydata, test_size = 0.2, random_state = 42)
    return np.array(x_train),np.array(x_validation),np.array(y_train),np.array(y_validation)

def load_testdata(path):
    x_test = []
    print 'test'
    validlist = []
    listing = os.listdir(path+'test\\')

    for r in listing:
        img1 = cv2.imread(path+'test\\'+r)
        if img1 is not None:
            img = cv2.resize(img1,(28,28))
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #### If you requires Grey images
            x_test.append(img)
        else:
            img = np.zeros((28, 28,3), dtype=float)
            x_test.append(img)
    x_test = np.array(x_test)
    imgname = map(lambda imgname: imgname.replace('.jpg', ''), listing)
    imgname = np.array(imgname)
    imgname = imgname.astype(np.int)
    ####  It will store numpy array onto provided location in "path"
    np.save(path + 'vggxtestcolor28.npy', x_test)
    np.save(path + 'vggimgnamecolor.npy', imgname)

    print 'test data set shape'
    print x_test.shape
    print imgname.shape
    return imgname,x_test


def main(args):
    path = 'C:\\Users\\Krunal\\Documents\\DSBA\\Spring 2016\\ML\\Project\\project\\'
    ############ Read CSV file to iterate images & load Y lables.
    traincsv = genfromtxt(path + 'train.csv', delimiter=',', skip_header=True)
    ############ Load Train Data
    X_train, X_validation, y_train, y_validation = load_data(path, traincsv.astype(int))
    ########### Load Test Data
    img_name,X_test=load_testdata(path)

if __name__ == '__main__':
    main(sys.argv)

