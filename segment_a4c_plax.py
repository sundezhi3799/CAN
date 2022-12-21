# example: python .\segment_a4c_plax.py -p 'I00610693534.dcm.jpg' -m 'unet_a4c_unet_20221110_17_02_47.hdf5'
from __future__ import division, print_function, absolute_import
import numpy as np
import os
from scipy.misc import imresize, imread, imsave
import tensorflow.compat.v1 as tf
from unet_a4c import Network
from unet_cardiac import Network as n1
from optparse import OptionParser

tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def segmentstudy(img, model):
    preds = np.argmax(model.predict(img)[0, :, :, :], 2)
    return preds


def load_model(view_n):
    feature_dim = 3
    drop_out = 1
    model_name = 'unet'
    if view_n == 'a4c':
        unet = Network(model_name, feature_dim, 5, drop_out).model
        model_n = 'unet_a4c_unet_20221110_17_02_47.hdf5'
        unet.load_weights('models/' + model_n)
    elif view_n == 'plax':
        unet = Network(model_name, feature_dim, 4, drop_out).model
        model_n = 'unet_plax_unet_20221110_17_01_22.hdf5'
        unet.load_weights('models/' + model_n)
    else:
        unet = n1(model_name, feature_dim, 2, drop_out).model
        model_n = 'unet_cardiac_unet_20221120_21_38_48.hdf5'
        unet.load_weights('models/' + model_n)
    return unet


def main(img, unet):
    image1 = np.array([imresize(img, (224, 224))]).astype('float32')
    image1 /= image1.max()
    preds = segmentstudy(np.array(image1), unet)
    return preds


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-p", "--imagepath", dest="imagepath", help="imagepath")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    parser.add_option("-v", "--view", dest="view", default="cardiac", help="view")
    params, args = parser.parse_args()
    image_path = params.imagepath
    view_n = params.view
    model_name = 'unet'
    
    image = imread(image_path)
    unet = load_model(view_n)
    fig = main(image, unet)
    image1 = np.array([imresize(image, (224, 224))])
    TF = fig == 0
    TF = TF[:, :, np.newaxis]
    TF = np.concatenate([TF, TF, TF], axis=2)
    image1[0][TF] = 0
    imsave('segmented.jpg', image1[0])
    imsave('masks.jpg', fig)
