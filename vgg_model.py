import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input



def load_data(train_or_test):
    """
    load labels(pandas['id','breed']) and images(np.array [N,H,W,C] )
    input parameter = 'train' or 'test'
    """
    path_dir = '../../data/dog-breed-identification/'

    if train_or_test == 'train':
        labels = pd.read_csv(path_dir+'labels.csv')
        nums = len(labels)
    elif train_or_test == 'test':
        labels = pd.read_csv(path_dir+'sample_submission.csv')
        nums = len(labels)

    imgs = np.zeros((nums,224,224,3),dtype='float32')
    for i , img_id in enumerate(labels['id']):
        path = path_dir + train_or_test + '/%s.jpg'%img_id
        img = image.img_to_array(image.load_img(path,target_size=(224,224)))
        imgs[i] = img/255.
    return labels, nums, imgs


def load_data_top(n):
    """
    load top n classes labels(pandas['id','breed'])
                           and images(np.array [N,H,W,C] )
    """
    path_dir = '../../data/dog-breed-identification/'

    labels = pd.read_csv(path_dir+'labels.csv')
    selected_breed = list(labels.groupby('breed').count().sort_values(by='id',ascending=False).index[:n])
    labels = labels[labels['breed'].isin(selected_breed)]
    nums = len(labels)

    imgs = np.zeros((nums,224,224,3),dtype='float32')
    for i , img_id in enumerate(labels['id']):
        path = path_dir + 'train/%s.jpg'%img_id
        img = image.img_to_array(image.load_img(path,target_size=(224,224)))
        imgs[i] = img/255.
    return labels, nums, imgs


def Pre_trained_vgg16net(width, height, depth, classes):
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16= VGG16(weights='imagenet', include_top=False)
    model_vgg16.summary()

    #Create your own input format (here 3x200x200)
    input = Input(shape=(width,height,depth),name = 'image_input')

    #Use the generated model
    output_vgg16= model_vgg16(input)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    #Create your own model
    my_model = Model(input=input, output=x)

    return my_model


def plot_history(history,to_file=None):
    plt.figure(figsize = (15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    if to_file is not None:
        plt.savefig(to_file,dpi=300)
    plt.show()
    return


if __name__ == '__main__':

    # loading training datasets
    print('* loading datasets ...')
    labels,train_nums, x_train = load_data_top(5)
    print('* number of datasets = ',train_nums)
    print('* datasets shape = ',x_train.shape)
    plt.imshow(x_train[0])


    # trainsform label to one-hot-encoder
    label_encoder = LabelEncoder()
    label_int = label_encoder.fit_transform(labels['breed'])
    y_train = np_utils.to_categorical(label_int)
    print('* training target shape = ',y_train.shape)
    print('* check #y_train = #x_train : ',y_train.shape[0]==x_train.shape[0])


    # validation
    random_seed = 2
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=random_seed)
    print('* training data shape = ',x_train.shape)
    print('* validation data shape = ',x_val.shape)


    # vgg16net model training
    print('*** VGG16Net model training ***')

    vgg16_model = Pre_trained_vgg16net(224,224,3,5)
    vgg16_model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    vgg16_model.summary()
    plot_model(vgg16_model, show_shapes=True,to_file='model.png')

    # training
    history = vgg16_model.fit(x_train, y_train, epochs=30, batch_size=32 , validation_data=(x_val,y_val))

    # saving model, history and history figure
    vgg16_model.save('./result/vgg16net_model.h5')
    plot_history(history,to_file='./result/vgg16net_model')
    with open('./result/history_vgg16net_model.h5', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)



