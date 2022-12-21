import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
from vgg import Network
import os, time
from scipy.misc import imread, imresize
import matplotlib.cm as cm
import pandas as pd
import segment_a4c_plax

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.disable_eager_execution()
model_builder = keras.applications.xception.Xception
img_size = (224, 224)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions


def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(sess, img_array, model):
    last_conv_layer_output, preds = model.conv_5_3, model.fc_8
    class_channel = tf.reduce_max(preds, axis=1)
    grads = tf.gradients(class_channel, last_conv_layer_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap_lst = []
    for i in range(img_array.shape[0]):
        heatmap = last_conv_layer_output[i] @ pooled_grads[i][..., tf.newaxis]
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = sess.run(tf.squeeze(heatmap), {model.input: img_array})
        heatmap_lst.append(heatmap)
    return np.array(heatmap_lst)


def attention_weight(heatmaps, masks):
    wrv = 0
    wlv = 0
    wra = 0
    wla = 0
    for k in range(len(heatmaps)):
        hk = imresize(heatmaps[k], (224, 224))
        mask = masks[k]
        wkh = sum(sum(hk))
        mk_rv = mask.copy()
        mk_rv[mask != 3] = 0
        mk_lv = mask.copy()
        mk_lv[mask != 1] = 0
        mk_ra = mask.copy()
        mk_ra[mask != 4] = 0
        mk_la = mask.copy()
        mk_la[mask != 2] = 0
        wkrv = sum(sum(mk_rv * hk))
        wklv = sum(sum(mk_lv * hk))
        wkra = sum(sum(mk_ra * hk))
        wkla = sum(sum(mk_la * hk))
        wrv += wkrv / wkh
        wlv += wklv / wkh
        wra += wkra / wkh
        wla += wkla / wkh
    s = wrv + wlv + wla + wra
    return wrv / s, wlv / s, wla / s, wra / s


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(heatmap * 255)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    jet_heatmap=keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap.save('jet'+cam_path)


def main(image_dir, labels_file_path, view):
    # Make model
    feature_dim = 3
    label_dim = 2
    learning_rate = 1e-4
    weight_decay = 5e-4
    sess = tf.Session()
    model = Network(weight_decay, learning_rate, feature_dim, label_dim, False)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables()[:32])
    if view == 'a4c':
        model_name = 'pah_2_class_a4c_20221026_13_02_32.ckpt'
    elif view == 'plax':
        model_name = 'pah_2_class_plax_20221027_17_08_06.ckpt'
    else:
        model_name = None
    model_path = os.path.join('models/', model_name)
    saver.restore(sess, model_path)
    labels_file = pd.read_csv(labels_file_path, index_col=0).iloc[:1000, ]
    studys = labels_file.index.tolist()
    names = []
    img_lst = []
    for study in studys:
        for image_name in eval(labels_file.loc[study, view]):
            y = labels_file.loc[study, 'PAH']
            if y:
                img_path = os.path.join(image_dir, image_name)
                image = imread(img_path).astype('uint8')
                image = imresize(image, (224, 224))
                if len(image.shape) == 3:
                    img = image
                    VGG_MEAN = [0, 0, 0]
                    # Convert RGB to BGR and subtract mean
                    img_array = (img - VGG_MEAN)
                    img_lst.append(img_array)
                    names.append(image_name)
    img_arrays = np.array(img_lst)
    time1 = time.time()
    heatmaps = []
    masks = []
    unet = segment_a4c_plax.load_model(view)
    for i in range(0, len(names)):
        batch_names = [names[i]]
        batch_imgs = img_arrays[i][np.newaxis, :, :]
        preds = np.argmax(model.predict(sess, batch_imgs), axis=1)
        if preds[0]:
            heatmap = make_gradcam_heatmap(sess, batch_imgs, model)
            mask = segment_a4c_plax.main(batch_imgs[0], unet)
            masks.append(mask)
            heatmaps.append(heatmap[0])
            outdir = 'hotmap_' + model_name
            os.makedirs(outdir, exist_ok=True)
            os.makedirs('jet' + outdir, exist_ok=True)
            for j in range(len(batch_names)):
                image_name = batch_names[j]
                img_path = os.path.join(image_dir, image_name)
                save_and_display_gradcam(img_path, heatmap[j],
                                         cam_path=os.path.join(outdir, str(preds[j]) + '_' + image_name))
    wrv, wlv, wla, wra = attention_weight(heatmaps, masks)
    print('Done!' + ' ' + str(time.time() - time1) + 's')
    return wrv, wlv, wla, wra


if __name__ == '__main__':
    wrv, wlv, wla, wra = main(image_dir="segmented-images-all",
                              labels_file_path="train_test_pah_a4c_plax_all.csv",
                              view='plax')
    print(wrv, wlv, wla, wra)
