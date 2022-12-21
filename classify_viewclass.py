# example: python .\predict_viewclass_v3_gpu.py -d  "D:\data\Echocardiography\\xizong\echo-jpg-2022-testid"
# -g 0 -m "view_6_resnet_resnet50_20221109_15_23_30.hdf5"
from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow.compat.v1 as tf
import os

from optparse import OptionParser
from scipy.misc import imread, imresize
from resnet_6_views import Network

tf.disable_v2_behavior()


def classify(images, modelname):
    """
    Classifies echo images with given model
    @param images: arrays with echo images for classification
    @param model: model for classification
    """
    infile = open("viewclasses.txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]
    feature_dim = 3
    label_dim = len(views)
    model_name = './models/' + modelname
    model = Network('resnet50', feature_dim, label_dim).model
    model.load_weights(model_name)
    predictions = np.around(model.predict(images), decimals=3)
    return predictions.argmax()


def main(root, modelname):
    # writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
    # writer.close()
    infile = open("viewclasses.txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 3
    label_dim = len(views)

    out = open(modelname + "_" + "probabilities.txt", 'w+')
    out.write("study\timage")
    for j in views:
        out.write("\t" + "prob_" + j)
    out.write('\n')
    model_name = './models/' + modelname
    model = Network('resnet50', feature_dim, label_dim).model
    model.load_weights(model_name)
    for testids in os.listdir(root):
        imagenames = []
        images = []
        imagedir = os.path.join(root, testids)
        for filename in os.listdir(imagedir):
            if "jpg" in filename:
                if len(imagenames) < 64:
                    image = imread(os.path.join(imagedir, filename)).astype('uint8')
                    images.append(imresize(image, (224, 224)))
                    imagenames.append(filename)
                else:
                    predictions = classify(np.array(images), model)
                    for i in range(len(imagenames)):
                        prefix = imagenames[i]
                        out.write(testids + "\t" + prefix)
                        for j in predictions[i]:
                            out.write("\t" + str(j))
                        out.write("\n")
                    imagenames = []
                    images = []
        if images:
            predictions = classify(np.array(images), model)
            for i in range(len(imagenames)):
                prefix = imagenames[i]
                out.write(testids + "\t" + prefix)
                for j in predictions[i]:
                    out.write("\t" + str(j))
                out.write("\n")
    out.close()


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-d", "--imagedir", dest="imagedir", help="imagedir")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    parser.add_option("-m", "--model", dest="model",default='view_6_resnet_resnet50_20221109_15_23_30.hdf5')
    params, args = parser.parse_args()
    image_dir = params.imagedir
    model_n = params.model
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    # function
    main(image_dir, model_n)
