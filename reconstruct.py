# example: python .\reconstruct.py -p "I00651076388.dcm.jpg" -m 'models\unet_a4c_unet_20221110_17_02_47.hdf5' -g 0 -v 'a4c'
from scipy.misc import imread, imresize, imsave
import os
import segment_a4c_plax
import unet_plax
from optparse import OptionParser


def main(image, model, weights):
    imager = 255 - image
    image_o = imresize(image, (224, 224))
    mask = segment_a4c_plax.main(image_o, model)
    fig3 = imager.copy()
    fig3 = fig3 / fig3.max()
    for n in weights.keys():
        fig3[mask == n] = fig3[mask == n] * weights[n]
    return mask, fig3


if __name__ == '__main__':
    # Hyperparams
    parser = OptionParser()
    parser.add_option("-p", "--imagepath", dest="image_path", help="images path")
    parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
    parser.add_option("-m", "--model_path", dest="model_path", help="model path")
    parser.add_option("-v", "--view", dest="view", help="view")
    params, args = parser.parse_args()
    image_path = params.image_path
    gpu = params.gpu
    model_path = params.model_path
    view = params.view
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    image_s = imread(image_path).astype('uint8')
    if view == 'a4c':
        unet = unet_plax.Network('unet', 3, 5, 0.5, False).model
        unet.load_weights(model_path)
        weight = {3: 0.468, 1: 401, 4: 0.127, 2: 0.004}
    else:
        unet = unet_plax.Network('unet', 3, 4, 0.5, False).model
        unet.load_weights(model_path)
        weight = {3: 0.937, 1: 0.052, 2: 0.011}
    masks, fig = main(image_s, unet, weight)
    imsave('reconstructed.png', fig)
    imsave('mask.png', masks)
