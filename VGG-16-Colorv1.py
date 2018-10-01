#########################################################################################
#Problem : Hotel Image Classification
#Training Set : 38,372 images
#Test Set : 19,648 images
#Author : Krunal Dholakia, kdholaki@uncc.edu
#Notes :-
############ Used Keras package for deep Learning
############ Explain model structure and desing in final report
############ Best Model in use - VGG111Grey - VGG_111Grey() It has 37 layers , follow similar structure like VGG 16 except 5 less layers
############ Grey images are giving more accuracy rather than color image for me
#########################################################################################

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import sys
import os
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping


######### 17 layers network for color images
def VGG_1(weight_path=None):
    print 'VGG1'
    # input image dimensions
    img_rows, img_cols = 28,28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid',input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    if os.path.isfile(path + 'VGG1_weights_color.h5'):
        print 'Weight used'
        model.load_weights(path + 'VGG1_weights_color.h5')

    return model

#### 17 Layer Network for Grey images
def VGG_1Grey():
    print 'VGG1Grey'
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(64, nb_conv, nb_conv, border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    if os.path.isfile(path + 'VGG1_weights_grey.h5'):
        print 'Weight used'
        model.load_weights(path + 'VGG1_weights_grey.h5')
    return model

########## 22 Layer Network
def VGG_11Grey():
    print 'VGG11Grey'
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 2
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(64, nb_conv, nb_conv,border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(128, nb_conv, nb_conv,border_mode='valid',input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    if os.path.isfile(path + 'VGG1_weights_grey11.h5'):
        print 'Weight used'
        model.load_weights(path + 'VGG1_weights_grey11.h5')
    return model

##########  31 Layers Network
def VGG_111Grey():
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
    '''
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    '''
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

############# 32 Layer Network
def VGG_2():
    print 'VGG2'
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 28, 28)))
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
    '''
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    '''
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

    if os.path.isfile(path + 'VGG2_weights1.h5'):
        print 'Weight used!!'
        model.load_weights(path + 'VGG2_weights1.h5')

    return model

########### VGGNET Network model - reference given in Read me file
def VGG_3():
    print 'VGG 3'
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,28,28)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    ########### Loading pretrained weights of IMAGENET Competition - Shared by author
    if os.path.isfile(path + 'vgg16_weights.h5'):
        print 'Weight used!!'
        model.load_weights(path + 'vgg16_weights.h5')

    return model

########### Load color train images into numpy array
def load_data(path,traincsv):
    xdata = []
    ydata = []
    x_train = []
    x_validation = []
    y_train = []
    y_validation = []
    xdata = np.load(path + 'vggxdatacolor28.npy')
    ydata = np.load(path + 'vggydatacolor28.npy')
    xdata = xdata.astype('float32')
    ydata = ydata.astype('float32')
    print 'train data set shape'
    print xdata.shape
    print 'label data set shape'
    print ydata.shape
    x_train, x_validation, y_train, y_validation = train_test_split(xdata, ydata, test_size = 0.2, random_state = 42)
    return np.array(x_train),np.array(x_validation),np.array(y_train),np.array(y_validation)

########### Load grey train images into numpy array
def load_data_grey(path,traincsv):
    xdata = []
    ydata = []
    x_train = []
    x_validation = []
    y_train = []
    y_validation = []
    xdata = np.load(path + 'xdata.npy')
    ydata = np.load(path + 'ydata.npy')
    print 'train data set shape'
    print xdata.shape
    print 'label data set shape'
    print ydata.shape
    ######## Split data into training and validation
    x_train, x_validation, y_train, y_validation = train_test_split(xdata, ydata, test_size = 0.2, random_state = 42)
    return np.array(x_train),np.array(x_validation),np.array(y_train),np.array(y_validation)


########### Load color test images into numpy array
def load_testdata(path):
    x_test = []
    print 'test'
    x_test= np.load(path + 'vggxtestcolor28.npy')
    imgname=np.load(path + 'vggimgnamecolor.npy')
    x_test = x_test.astype('float32')
    #x_test -= np.mean(x_test,axis=2)
    #x_test -= np.std(x_test, axis=0)
    print 'test data set shape'
    print x_test.shape
    print imgname.shape
    return imgname,x_test

########### Load grey  test images into numpy array
def load_testdata_grey(path):
    x_test = []
    print 'test'
    x_test= np.load(path + 'xtest.npy') #### Store all ~19800 image data into (19800,28,28,3) shape numpy array
    imgname=np.load(path + 'vggimgnamecolor.npy') #### store image name of test data folder
    imgname = np.array(imgname)
    imgname = imgname.astype(np.int)
    print 'test data set shape'
    print x_test.shape
    print imgname.shape
    return imgname,x_test  ###### Return test image name and Test data


if __name__ == "__main__":
    path = '//home//ubuntu//notebook//CNN//' # Path of AWS Ubuntu location
    modelflag='VGG111Grey' # Created various Deep networks , intiated that network based on this flag
    iscolor = 'grey' # Need to decide which type of image to use - Just grey or BGR
    # input image dimensions
    img_rows, img_cols = 28, 28
    #batchsize
    batch_size = 4
    # Number of iteration for backpropogations
    nb_epoch = 6

    ############################# For Grey Input images ###############################
    if iscolor == 'grey':
        traincsv = genfromtxt(path + 'train.csv', delimiter=',', skip_header=True) # Read from CSV file
        ###### Load data into Train and validation numpy array for further computation
        X_train, X_validation, y_train, y_validation = load_data_grey(path, traincsv.astype(int))
        ###### Convert Data into (number of images, Channel , Number of rows -  i.e 28 , Number of cols  -  i.e 28  )
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
        ###### Convert numpy int into float
        X_train = X_train.astype('float32')
        X_validation = X_validation.astype('float32')

        ###### Below image preprocessing code is used once but Image preprocessing was decreasing accuracy
        '''
        datagen = ImageDataGenerator(featurewise_center=True,
                                     samplewise_center=False,
                                     featurewise_std_normalization=True,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     rotation_range=0.,
                                     width_shift_range=0.,
                                     height_shift_range=0.,
                                     shear_range=0.,
                                     zoom_range=0.,
                                     channel_shift_range=0.,
                                     fill_mode='nearest',
                                     cval=0.,
                                     horizontal_flip=False,
                                     vertical_flip=False,
                                     dim_ordering='tf')
        '''
        # Devide by 255 for normalization
        X_train /= 255
        X_validation /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_validation.shape[0], 'validation samples')

        ################## Train the model and save the wieght ##################
        if modelflag == 'VGG111Grey':
            print 'VGG111Grey'

            model = VGG_111Grey()  # Return an object "model" having entire Deep Network structure
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  #
            ###### Will compile model - Keras in built function
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            ###### parameter to set early stoping if validation accuracy goes down continueously for 2 iteration
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            ###### Keras In built function to for model training
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                      validation_data=(X_validation, y_validation), callbacks=[early_stopping])
            ###### Save trained model weight
            model.save_weights(path + 'VGG1_weights_grey111.h5', overwrite=True)
            score = model.evaluate(X_validation, y_validation, verbose=0)
            print('Validation score:', score[0])
            print('Validation accuracy:', score[1])


        if modelflag == 'VGG11Grey':
            print 'VGG11Grey'
            model = VGG_11Grey() # Return an object "model" having entire Deep Network structure
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) #
            ###### compile model - Keras in built function
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            ###### parameter to set early stoping if validation accuracy goes down continueously for 2 iteration
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            ###### Keras' in built function for model training
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                      validation_data=(X_validation, y_validation),callbacks=[early_stopping])
            ###### Save trained model weight
            model.save_weights(path + 'VGG1_weights_grey11.h5', overwrite=True)
            score = model.evaluate(X_validation, y_validation, verbose=0)
            print('Validation score:', score[0])
            print('Validation accuracy:', score[1])


        #################  Load Testing Data  ###########
        img_name, X_test = load_testdata_grey(path)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_test = X_test.astype('float32')
        #X_test -= np.mean(X_test, axis=0)
        X_test /= 255
        #datagen.fit(X_test)
        print(X_test.shape[0], 'test samples')

        if modelflag == 'VGG1Grey':
            print 'VGG1Grey'
            model = VGG_1Grey()
            sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            # datagen.fit(X_train)
            # datagen.fit(X_validation)
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                      validation_data=(X_validation, y_validation), callbacks=[early_stopping])
            model.save_weights(path + 'VGG1_weights_grey.h5', overwrite=True)
            score = model.evaluate(X_validation, y_validation, verbose=0)
            print('Validation score:', score[0])
            print('Validation accuracy:', score[1])

        #################  Load Grey Testing Images  ###########
        img_name, X_test = load_testdata_grey(path)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_test = X_test.astype('float32')
        # X_test -= np.mean(X_test, axis=0)
        X_test /= 255
        # datagen.fit(X_test)
        print(X_test.shape[0], 'test samples')

        ############################# Color Input images ###############################
        if iscolor == 'color':
            ########## Load Training Data ####################
            traincsv = genfromtxt(path + 'train.csv', delimiter=',', skip_header=True)
            X_train, X_validation, y_train, y_validation = load_data(path, traincsv.astype(int))
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_validation = X_validation.reshape(X_validation.shape[0], 3, img_rows, img_cols)
            X_train = X_train.astype('float32')
            X_validation = X_validation.astype('float32')
            # X_train -= np.mean(X_train, axis=0)
            # X_validation -= np.mean(X_validation, axis=0)
            X_train /= 255
            X_validation /= 255
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_validation.shape[0], 'validation samples')
            ################## Train the model and save the wiegt ##################

            if modelflag == 'VGG1':
                print 'VGG1'
                model = VGG_1()
                model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
                model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                          validation_data=(X_validation, y_validation))
                model.save_weights(path + 'VGG1_weights_color.h5', overwrite=True)
                score = model.evaluate(X_validation, y_validation, verbose=0)
                print('Validation score:', score[0])
                print('Validation accuracy:', score[1])

            if modelflag == 'VGG2':
                model = VGG_2()
                model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=0)
                model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                          validation_data=(X_validation, y_validation), callbacks=[early_stopping])
                model.save_weights(path + 'VGG2_weights1.h5', overwrite=True)
                score = model.evaluate(X_validation, y_validation, verbose=0)
                print('Validation score:', score[0])
                print('Validation accuracy:', score[1])

            if modelflag == 'VGG3':
                model = VGG_3()
                model.layers.pop()
                model.add(Dense(8, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                          validation_data=(X_validation, y_validation), callbacks=[early_stopping])
                model.save_weights(path + 'vgg16_weights.h5', overwrite=True)
                score = model.evaluate(X_validation, y_validation, verbose=0)
                print('Validation score:', score[0])
                print('Validation accuracy:', score[1])

            #################  Load COLOR Testing Data  ###########
            img_name, X_test = load_testdata(path)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            X_test = X_test.astype('float32')
            # X_test -= np.mean(X_test,axis=0)
            X_test /= 255
            print(X_test.shape[0], 'test samples')

    ################# Predict the output probablities ##############
    output = model.predict_proba(X_test, batch_size=32, verbose=1)
    y_test = np.column_stack((img_name, output)).astype(np.float)
    np.savetxt(path + 'test.csv', y_test, header="id,col1,col2,col3,col4,col5,col6,col7,col8", comments='',
               delimiter=',', fmt='%i,%f,%f,%f,%f,%f,%f,%f,%f')