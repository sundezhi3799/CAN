# example: python resnet_a4c.py -i '../data/segmented-images-all' -l '../xizong/train_test_pah_a4c_plax_all.csv'
import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Model, callbacks
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imresize
import random
import os, datetime, pickle
import numpy as np
import pandas as pd
import json
from optparse import OptionParser

tf.disable_v2_behavior()


# # Network
class Network(object):
    def __init__(self, feature_dim=1, label_dim=2, dropout=0.5):
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.model = self.resnet50()

    def resnet50(self):
        base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
        # base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.label_dim, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
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


def load_train_test_data(image_dir, labels_file, view, label_dim):
    res = []
    studynum = 0
    positive_study = 0
    positive_img = 0
    neg_img = 0
    positive_train_study = 0
    train = 1
    train_index = random.sample(list(labels_file.index), int(len(labels_file.index) * 0.8))
    test_index = labels_file.index.difference(train_index)
    for sample_index in [train_index, test_index]:
        labels_list = []
        imagelist = []
        for study in sample_index:
            studynum += 1
            y = labels_file.loc[study, 'PAH']
            if y:
                if train:
                    positive_train_study += 1
                positive_study += 1
            viewpoint = eval(labels_file.loc[study, view])
            for image_name in viewpoint:
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


def main(image_dir, labels_file, view, label_dim, feature_dim, batch_size, epochs, learning_rate, weight_decay,
         momentum,
         drop_out):
    x_train, y_train, x_test, y_test = load_train_test_data(image_dir, labels_file, view, label_dim)
    early_stop = False
    reducing_lr = False
    save_best = False
    par_dct = {'feature_dim': feature_dim, 'label_dim': label_dim, 'batch_size': batch_size,
               'epochs': epochs, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
               'momentum': momentum, 'drop_out': drop_out, 'early_stop': early_stop, 'reducing_lr': reducing_lr,
               'save_best': save_best}
    network = Network(feature_dim, label_dim, drop_out)
    model_name = 'resnet50'
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


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-i", "--imagedir", dest="imagedir", help="images dir")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    parser.add_option("-l", "--labelfile", dest="labelfile", help="labels file")
    parser.add_option("-v", "--view", dest="view", help="view")
    parser.add_option("-n", "--label_dim", dest="label_dim", default=5, type='int')
    parser.add_option("-f", "--feature_dim", dest="feature_dim", default=3, type='int')
    parser.add_option("-b", "--batch_size", dest="batch_size", default=64, type='int')
    parser.add_option("-e", "--epochs", dest="epochs", default=100, type='int')
    parser.add_option("-r", "--learning_rate", dest="learning_rate", default=1e-3, type='float')
    parser.add_option("-w", "--weight_decay", dest="weight_decay", default=5e-4, type='float')
    parser.add_option("-m", "--momentum", dest="momentum", default=0.9, type='float')
    parser.add_option("-d", "--drop_out", dest="drop_out", default=0.5, type='float')
    params, args = parser.parse_args()
    image_dir = params.imagedir
    gpu = params.gpu
    labels_file = params.labelfile
    view = params.view
    label_dim = params.label_dim
    feature_dim = params.feature_dim
    batch_size = params.batch_size
    epochs = params.epochs
    learning_rate = params.learning_rate
    weight_decay = params.weight_decay
    momentum = params.momentum
    drop_out = params.drop_out
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    labels_table = pd.read_csv(labels_file, index_col=0)
    main(image_dir, labels_table, view, label_dim, feature_dim, batch_size, epochs, learning_rate, weight_decay,
         momentum,
         drop_out)
