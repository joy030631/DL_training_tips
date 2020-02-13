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


def CNN(width, height, depth, classes):
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (width, height, depth)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(units=256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=classes, activation = "softmax"))

    return model


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


    ## cnn model training (the difference between different batch size)
    print('*** CNN model training ***')

    for n in [128,64,32,16,8,1]:
        print('* batch size = ',n)

        #cnn model
        cnn_model = CNN(224,224,3,5)
        cnn_model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        cnn_model.summary()
        cnn_model.load_weights('./result/initial_weights.h5')

        # training
        history = cnn_model.fit(x_train, y_train, epochs=100, batch_size=n , validation_data=(x_val,y_val))

        # saving model, history and history figure
        cnn_model.save('./result/cnn_model_batch%s.h5'%n)
        plot_history(history,to_file='./result/cnn_model_batch%s'%n)
        with open('./result/history_cnn_model_batch%s.pkl'%n, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)



