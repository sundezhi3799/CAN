import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Sequential, Model, callbacks, regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception
from scipy.misc import imread, imresize
import os, datetime, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random, json
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# # Network
class Network(object):
    def __init__(self, model_name='inceptionv3', feature_dim=1, label_dim=2, dropout=0.5, maxout=False):
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        print('model: ', model_name)
        if model_name == 'inceptionv3':
            self.model = self.inceptionv3()
        if model_name == 'resnet50':
            self.model = self.resnet50()
        if model_name == 'densenet121':
            self.model = self.densenet121()
        if model_name == 'xception':
            self.model = self.xception()

    def inceptionv3(self):
        base_model = InceptionV3(include_top=False, weights=None, input_shape=(224, 224, 3))
        model = Sequential()
        model.add(GlobalAveragePooling2D(name='avg_pool_last'))
        model.add(Dense(1024, activation='relu', name='fc1'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.label_dim, activation='softmax', name='prediction'))
        model = Model(inputs=base_model.input, outputs=model(base_model.output))
        return model

    def resnet50(self):
        base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.label_dim, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def densenet121(self):
        base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)  # 对数据进行正则化
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization(name='bn_fc_01')(x)
        predictions = Dense(self.label_dim, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def xception(self):
        base_model = Xception(include_top=False, weights=None, input_shape=(224, 224, 3))
        model = self.top_model(base_model)
        model = Model(inputs=base_model.input, outputs=model(base_model.output))
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


def load_train_test_data(label_dim, view):
    labels_file = pd.read_csv('train_test_pah_a4c_plax_all.csv', index_col=0).iloc[:2000, :]
    train_index = pd.read_csv('train_index.csv', header=None)[0].tolist()
    test_index = pd.read_csv('test_index.csv', header=None)[0].tolist()
    res = []
    studynum = 0
    positive_study = 0
    positive_img = 0
    neg_img = 0
    positive_train_study = 0
    train = 1
    for sample_index in [train_index, test_index]:
        labels_list = []
        imagelist = []
        for image_dir in ['segmented-images-all']:
            for study in sample_index:
                studynum += 1
                y = labels_file.loc[study, 'PAH']
                if y:
                    if train:
                        positive_train_study += 1
                    positive_study += 1
                viewpoint = eval(labels_file.loc[study, view])
                for lst in [viewpoint]:
                    for image_name in lst:
                        image = imread(os.path.join(image_dir, image_name)).astype('uint8')
                        label = [0] * label_dim
                        if y:
                            imagelist.append(imresize(image, (224, 224)))
                            positive_img += 1
                            label[1] = 1
                            labels_list.append(label)
                        else:
                            imagelist.append(imresize(image, (224, 224)))
                            label[0] = 1
                            labels_list.append(label)
                            neg_img += 1
        res.append([imagelist, labels_list])
        train = 0
    print('\nstudy number: ', studynum, '\npos study: ', positive_study, '\nneg img: ', neg_img, '\npos img: ',
          positive_img, '\ntrain pos study: ', positive_train_study)
    return np.array(res[0][0]), np.array(res[0][1]), np.array(res[1][0]), np.array(res[1][1])


if __name__ == '__main__':
    label_dim = 2
    view = 'a4c'
    x_train, y_train, x_test, y_test = load_train_test_data(label_dim, view)
    for model_name in ['resnet50', 'inceptionv3', 'densenet121', 'xception']:
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
        model_name = model_name
        par_dct = {'feature_dim': feature_dim, 'label_dim': label_dim, 'batch_size': batch_size,
                   'epochs': epochs, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
                   'momentum': momentum, 'drop_out': drop_out, 'early_stop': early_stop, 'reducing_lr': reducing_lr,
                   'save_best': save_best, 'model_name': model_name}
        network = Network(model_name, feature_dim, label_dim, drop_out, False)
        kf = KFold(5, shuffle=True, random_state=33)
        timenow = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        out_path = "model/pah_2_%s_%s_%s.hdf5" % (view, model_name, timenow)
        history = network.train(out_path, learning_rate, weight_decay, momentum, epochs, batch_size, x_train, y_train,
                                x_test, y_test, early_stop=early_stop, reducing_lr=reducing_lr, save_best=save_best)
        os.makedirs('parameter', exist_ok=True)
        json.dump(par_dct, open('parameter//pah_2_%s_%s_%s.json' % (view, model_name, timenow), 'w+'))
        if not save_best:
            os.makedirs('model', exist_ok=True)
            network.model.save_weights("model/pah_2_%s_%s_%s.hdf5" % (view, model_name, timenow))
            print("model/pah_2_%s_%s_%s.hdf5" % (view, model_name, timenow))
        os.makedirs('history', exist_ok=True)
        with open("history/pah_2_%s_%s_%s.pickle" % (view, model_name, timenow), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
