import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Sequential, Model, callbacks
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Input, GlobalAveragePooling2D, \
    BatchNormalization, AveragePooling2D, Conv2D, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from scipy.misc import imread, imresize
import os, datetime, pickle
import numpy as np
import pandas as pd
import random, json
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import layers

tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# # Network
class Network(object):
    def __init__(self, feature_dim=1, label_dim=2, dropout=0.5):
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.model = self.resnet50()

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
            x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2,
                                                                                    2)):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(
            filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
            input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
            x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(
            filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
            input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def base_model(self, include_top=False,
                   input_shape=None,
                   pooling=None,
                   classes=6):
        img_input = Input(shape=input_shape)
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(
            64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        if include_top:
            x = Flatten()(x)
            x = Dense(classes, activation='softmax', name='fc1000')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnet50')
        return model

    def resnet50(self):
        base_model = self.base_model(include_top=False, input_shape=(224, 224, self.feature_dim))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.label_dim, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.summary()
        return model

    def top_model(self, base_model):
        model = Sequential()
        model.add(Flatten(name='flatten', input_shape=(base_model.output_shape[1:])))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dropout(self.dropout))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.label_dim, activation='softmax', name='prediction'))
        return model

    def train(self, out_path, learning_rate, weight_decay, momentum, epochs, batch_size, x_train, y_train, x_test,
              y_test, loss='binary_crossentropy', metrics=['accuracy'], early_stop=True, reducing_lr=True,
              save_best=True):
        checkpoint = ModelCheckpoint(out_path, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        self.model.compile(loss=loss, optimizer=SGD(decay=weight_decay, learning_rate=learning_rate, momentum=momentum),
                           metrics=metrics)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
        loss_callback = callbacks.EarlyStopping(monitor='loss', patience=5)
        callbacks_list = []
        if early_stop:
            callbacks_list.append(loss_callback)
        if reducing_lr:
            callbacks_list.append(reduce_lr)
        if save_best:
            callbacks_list.append(checkpoint)
        history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                 batch_size=batch_size, verbose=1, callbacks=callbacks_list)
        return history

    def test(self):
        score = self.model.evaluate(x_test, y_test, verbose=1)
        print("Large CNN Error: %.2f%%" % (100 - score[1] * 100))


def upsample(x, y):
    img_lst = []
    lab_lst = []
    for img, lab in zip(x, y):
        if np.argmax(lab) == 1:
            for i in range(8):
                img_lst.append(img)
                lab_lst.append(lab)
        img_lst.append(img)
        lab_lst.append(lab)
    index = list(range(len(img_lst)))
    random.shuffle(index)
    return np.array(img_lst)[index], np.array(lab_lst)[index]


def load_train_test_data(label_dim):
    labels_file = pd.read_csv('views.csv', index_col=0)
    data = pd.DataFrame()
    for i in range(label_dim):
        datai = labels_file.loc[labels_file['label'] == i]
        data = pd.concat([data, datai])
    image_names = data['image'].tolist()
    image_dir = '/home/administrator/data/echo-jpg-2022-images'
    labels_list = []
    imagelist = []
    for image_name in image_names:
        image = imread(os.path.join(image_dir, image_name)).astype('uint8')
        imagelist.append(imresize(image, (224, 224)))
        label = [0] * label_dim
        y = int(data.loc[data['image'] == image_name, 'label'].values[0])
        label[y] = 1
        labels_list.append(label)
    return np.array(labels_list), np.array(imagelist)


if __name__ == '__main__':
    label_dim = 6
    labels, images = load_train_test_data(label_dim)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=33)
    feature_dim = 3
    batch_size = 64
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 5e-4
    momentum = 0.9
    drop_out = 0.5
    early_stop = False
    reducing_lr = False
    save_best = False
    model_name = 'resnet50'
    par_dct = {'feature_dim': feature_dim, 'label_dim': label_dim, 'batch_size': batch_size,
               'epochs': epochs, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
               'momentum': momentum, 'drop_out': drop_out, 'early_stop': early_stop, 'reducing_lr': reducing_lr,
               'save_best': save_best}
    network = Network(feature_dim, label_dim, drop_out)
    timenow = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    out_path = "model/view_6_resnet_%s_%s.hdf5" % (model_name, timenow)
    history = network.train(out_path, learning_rate, weight_decay, momentum, epochs, batch_size, x_train, y_train,
                            x_test, y_test, early_stop=early_stop, reducing_lr=reducing_lr, save_best=save_best)
    os.makedirs('parameter', exist_ok=True)
    json.dump(par_dct, open('parameter//view_6_resnet_%s_%s.json' % (model_name, timenow), 'w+'))
    if not save_best:
        os.makedirs('model', exist_ok=True)
        network.model.save_weights("model/view_6_resnet_%s_%s.hdf5" % (model_name, timenow))
        print("model/view_6_resnet_%s_%s.hdf5" % (model_name, timenow))
    os.makedirs('history', exist_ok=True)
    with open("history/view_6_resnet_%s_%s.pickle" % (model_name, timenow), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
