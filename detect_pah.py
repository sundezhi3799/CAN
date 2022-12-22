import os
from optparse import OptionParser
from scipy.misc import imread, imresize
from train_resnet_pah import Network
import reconstruct as rt
import unet_a4c_plax
import predict_viewclass
import unet_cardiac
import consensus_voting


def image_level(image, unet, weight, model):
    masks, fig = rt.main(image, unet, weight)
    feature_dim = 3
    label_dim = 2
    resnet = Network(feature_dim, label_dim, dropout=1).model
    resnet.load_weights(model)
    prediction = resnet.predict(fig)[1]
    return prediction


def main(image_dir):
    a4c_scores = []
    plax_scores = []
    for image_name in os.listdir(image_dir):
        image = imread(os.path.join(image_dir, image_name)).astype('uint8')
        image = imresize(image, (224, 224))
        view = predict_viewclass.classify([image], 'view_6_resnet_resnet50_20221109_15_23_30.hdf5')
        image = unet_cardiac.segment(image, 'unet_cardiac_unet_20221120_21_38_48.hdf5')
        if view == 1:
            model = "pah_2_a4c_resnet50_20221108_15_55_59.hdf5"
            unet = unet_a4c_plax.Network('unet', 3, 5, 0.5, False).model
            unet.load_weights('unet_a4c_unet_20221110_17_02_47.hdf5')
            weight = {3: 0.37 / 0.37, 1: 0.35 / 0.37, 4: 0.20 / 0.37, 2: 0.08 / 0.37}
            a4c_scores.append(image_level(image, unet, weight, model))
        elif view == 2:
            model = "pah_2_plax_resnet50_20221108_16_00_52.hdf5"
            unet = unet_a4c_plax.Network('unet', 3, 4, 0.5, False).model
            unet.load_weights('unet_plax_unet_20221110_17_01_22.hdf5')
            weight = {3: 1 - 0.88, 1: 1 - 0.09, 2: 1 - 0.03}
            plax_scores.append(image_level(image, unet, weight, model))
        else:
            continue
    diag = consensus_voting.voting(a4c_scores, plax_scores)
    return diag


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-d", "--image_dir", dest="image dir", help="image dir")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    params, args = parser.parse_args()
    image_dir = params.image_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    diag = main(image_dir)
    print(diag)
