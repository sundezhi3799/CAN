import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Sequential, Model, callbacks, regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Input, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imresize
import os, datetime, pickle
import numpy as np
import pandas as pd
import random, json
from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# # Network
class Network(object):
    def __init__(self, feature_dim=1, label_dim=2, dropout=0.5):
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.model = self.resnet50()

    def resnet50(self):
        base_model = ResNet50(include_top=False, input_shape=(224, 224, self.feature_dim))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.label_dim, activation='softmax')(x)
        # 训练模型
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
        # 有一次提升, 则覆盖一次.
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
