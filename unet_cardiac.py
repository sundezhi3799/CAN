import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.optimizers import SGD
import os, datetime, pickle
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import KFold
from optparse import OptionParser

tf.disable_v2_behavior()


# # Network
class Network(object):
    def __init__(self, model_name='unet', feature_dim=3, label_dim=2, dropout=0.5):
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        print('model: ', model_name)
        if model_name == 'unet':
            self.model = self.unet()

    def unet(self):
        # base_model.summary()
        inputs = Input((224, 224, 3))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(self.label_dim, 1, activation='sigmoid')(conv9)
        model = Model(inputs, conv10)
        # model.summary()
        # writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
        # writer.close()
        return model

    def train(self, learning_rate, weight_decay, momentum, epochs, batch_size, x_train, y_train, x_test, y_test,
              loss='binary_crossentropy', metrics=['accuracy'], early_stop=True, reducing_lr=True):
        self.model.compile(loss=loss, optimizer=SGD(decay=weight_decay, learning_rate=learning_rate, momentum=momentum),
                           metrics=metrics)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
        loss_callback = callbacks.EarlyStopping(monitor='loss', patience=5)
        callbacks_list = []
        if early_stop:
            callbacks_list.append(loss_callback)
        if reducing_lr:
            callbacks_list.append(reduce_lr)
        history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                 batch_size=batch_size, verbose=1, callbacks=callbacks_list)
        return history


def data_preprocess(label, class_num):
    new_label = np.zeros(label.shape + (class_num,))
    for i in range(class_num):
        new_label[label == i, i] = 1
    label = new_label
    return label


def load_train_test_data(image_dir, labels_dir, label_dim):
    labels_list = []
    imagelist = []
    for image_name in os.listdir(labels_dir):
        label_name = os.path.splitext(image_name)[0] + '.png'
        image = imread(os.path.join(image_dir, image_name.replace('png', 'jpg'))).astype('float32')
        img = imresize(image, (224, 224))
        img = img / img.max()
        imagelist.append(img)
        label = imread(os.path.join(labels_dir, label_name), flatten=True).astype('float32')
        lab = imresize(label, (224, 224))
        lab = lab // lab.max()
        lst = list(set(lab.flatten().tolist()))
        for v in lst:
            lab[lab == v] = lst.index(v)
        lab1 = data_preprocess(lab, label_dim)
        labels_list.append(lab1)
    return np.array(labels_list), np.array(imagelist)


def main(image_dir, lables_dir, label_dim=2, feature_dim=3, batch_size=64, epochs=100, lr=1e-3, wd=5e-4, momentum=0.9,
         drop_out=0.5):
    early_stop = False
    reducing_lr = False
    model_name = 'unet'
    network = Network(model_name, feature_dim, label_dim, drop_out)
    labels, images = load_train_test_data(image_dir, lables_dir, label_dim)
    kf = KFold(100, shuffle=True, random_state=33)
    for train_index, test_index in kf.split(images):
        x_train = images[train_index]
        x_test = images[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        history = network.train(lr, wd, momentum, epochs, batch_size, x_train, y_train, x_test,
                                y_test, early_stop=early_stop, reducing_lr=reducing_lr)
        break
    timenow = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    network.model.save_weights("model/unet_cardiac_%s_%s.hdf5" % (model_name, timenow))
    os.makedirs('history', exist_ok=True)
    with open("history/unet_cardiac_%s_%s.pickle" % (model_name, timenow), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def segment(image, model_path):
    image = imresize(image, (224, 224))
    model = Network(model_name='unet').model
    model.load_weights(model_path)
    result = model.predict(np.array([image]))[0]
    image[result <= 0.95] = 0
    return image


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-i", "--imagedir", dest="imagedir", help="images dir")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    parser.add_option("-l", "--labeldir", dest="labeldir", help="labels dir")
    parser.add_option("-n", "--label_dim", dest="label_dim", default=2, type='int')
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
    label_dir = params.labeldir
    label_dim = params.label_dim
    feature_dim = params.feature_dim
    batch_size = params.batch_size
    epochs = params.epochs
    learning_rate = params.learning_rate
    weight_decay = params.weight_decay
    momentum = params.momentum
    drop_out = params.drop_out
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    main(image_dir, label_dir, label_dim, feature_dim, batch_size, epochs, learning_rate, weight_decay, momentum,
         drop_out)
