#image stuff
import cv2
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat

#the usual data science stuff
import os,sys
from glob import glob
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#keras
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout,Activation, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

#misc
import itertools

def vgg_face():
    img = Input(shape=(224, 224,3))

    #convolution layers
    conv1_1 = Conv2D(64, (3,3), activation='relu', name='conv1_1',padding='same')(img)
    conv1_2 = Conv2D(64, (3,3), activation='relu', name='conv1_2',padding='same')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3,3), activation='relu', name='conv2_1',padding='same')(pool1)
    conv2_2 = Conv2D(128, (3,3), activation='relu', name='conv2_2',padding='same')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3,3), activation='relu', name='conv3_1',padding='same')(pool2)
    conv3_2 = Conv2D(256, (3,3), activation='relu', name='conv3_2',padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3,3), activation='relu', name='conv3_3',padding='same')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3,3), activation='relu', name='conv4_1',padding='same')(pool3)
    conv4_2 = Conv2D(512, (3,3), activation='relu', name='conv4_2',padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3,3), activation='relu', name='conv4_3',padding='same')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3,3), activation='relu', name='conv5_1',padding='same')(pool4)
    conv5_2 = Conv2D(512, (3,3), activation='relu', name='conv5_2',padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3,3), activation='relu', name='conv5_3',padding='same')(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name = 'pool5')(conv5_3)

    #classification layer of original mat file
    fc6 = Conv2D(4096, (7,7), activation='relu', name='fc6',padding='valid')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096, (1,1), activation='relu', name='fc7',padding='valid')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    fc8 = Conv2D(2622, (1,1), activation='relu', name='fc8',padding='valid')(fc7_drop)
    flat = Flatten(name='flat')(fc8)
    prob = Activation('softmax',name='prob')(flat)

    model = Model(inputs=img, outputs=prob)

    return model

#Adapted from https://github.com/mzaradzki/neuralnets/tree/master/vgg_faces_keras
def copy_mat_to_keras(kmodel,l,noTop=True):
    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'
    if noTop:
        n=6
    else:
        n=0
    for i in range(l.shape[1]-n):
        mattype = l[0,i][0,0].type[0]
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            #skip layers without weights
            if len(kmodel.layers[kindex].get_weights())==0:
                continue
                
            #show details
            print(matname, mattype)
            print(l[0,i][0,0].weights[0,0].transpose(prmt).shape, l[0,i][0,0].weights[0,1].shape)
            print(kmodel.layers[kindex].get_weights()[0].shape, kmodel.layers[kindex].get_weights()[1].shape)
            print('------------------------------------------')
            
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
            
def predict_face(im,model,description):
    #convert to array
    arr = np.array(im).astype(np.float32)
    #add extra dimension
    arr = np.expand_dims(arr, axis=0)
    #mean center
    #arr[:,:,0] -= 129.1863
    #arr[:,:,1] -= 104.7624
    #arr[:,:,2] -= 93.5940
    #predict
    out = model.predict(arr)
    #get index
    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index,0]
    return(best_index, best_name[0], out[0,best_index])

#Stolen from fast.ai's Practical Deep Learning For Coders, Part 1 (http://course.fast.ai)
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=4, class_mode='categorical',target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

#Stolen from fast.ai's Practical Deep Learning For Coders, Part 1 (http://course.fast.ai)
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def predict_gender(im,model,label_name):
    #convert to array
    arr = np.array(im).astype(np.float32)
    #add extra dimension
    arr = np.expand_dims(arr, axis=0)
    #mean center
    #arr[:,:,0] -= 129.1863
    #arr[:,:,1] -= 104.7624
    #arr[:,:,2] -= 93.5940
    #predict
    out = model.predict(arr)
    #get index
    best_index = np.argmax(out, axis=1)[0]
    best_name = label_name[best_index]
    return(best_index, best_name, out[0,best_index])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
   