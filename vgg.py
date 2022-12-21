import numpy as np
import os
import nn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# # Network
class Network(object):
    def __init__(self, weight_decay, learning_rate, feature_dim=1, label_dim=8, maxout=False):
        self.x_train = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_train = tf.placeholder(tf.uint8, [None, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_test = tf.placeholder(tf.uint8, [None, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.network(self.x_train)  # network
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                           labels=self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.train_pred = self.network(self.x_train, keep_prob=1.0, reuse=True)  # network_1
        self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.train_pred, 1),
                                                              tf.argmax(self.y_train, 1)), tf.float32))
        self.val_pred = self.network(self.x_test, keep_prob=1.0, reuse=True)  # network_2
        self.val_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.val_pred, 1),
                                                           tf.argmax(self.y_test, 1)), tf.float32))

        self.probability = tf.nn.softmax(self.network(self.x_test, keep_prob=1.0, reuse=True))  # network_)3

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)

    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary),
                                         feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary

    # Training Accuracy
    def train_validate(self, sess, x_train, y_train):
        train_accuracy, train_summary = sess.run((self.train_accuracy, self.train_summary),
                                                 feed_dict={self.x_train: x_train, self.y_train: y_train})
        return train_accuracy, train_summary

    # Validation Accuracy
    def validate(self, sess, x_test, y_test):
        val_accuracy = sess.run((self.val_accuracy), feed_dict={self.x_test: x_test, self.y_test: y_test})
        val_loss = sess.run(self.loss, feed_dict={self.x_train: x_test, self.y_train: y_test})
        return val_accuracy, val_loss

    def predict(self, sess, x):
        '''
        Forward pass of the neural network. Predicts labels for images x.

        @params x: Numpy array of training images
        '''
        prediction = sess.run((tf.nn.softmax(self.val_pred)), feed_dict={self.x_test: x})
        return prediction

    def probabilities(self, sess, x):
        probability = sess.run((self.probability), feed_dict={self.x_test: x})
        return probability

    def network(self, input, keep_prob=0.5, reuse=None):
        with tf.variable_scope('network', reuse=reuse):
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            max_out_ = lambda x: nn.max_out(x, 16)
            conv_ = lambda x, output_depth, name, trainable=True: nn.conv(x, 3, output_depth, 1, self.weight_decay,
                                                                          name=name, trainable=trainable)
            fc_ = lambda x, features, name, relu=True: nn.fc(x, features, self.weight_decay, name, relu=relu,
                                                             trainable=True)
            self.input = input
            self.conv_1_1 = conv_(self.input, 64, 'conv1_1', trainable=True)
            self.conv_1_2 = conv_(self.conv_1_1, 64, 'conv1_2', trainable=True)

            self.pool_1 = pool_(self.conv_1_2)

            self.conv_2_1 = conv_(self.pool_1, 128, 'conv2_1', trainable=True)
            self.conv_2_2 = conv_(self.conv_2_1, 128, 'conv2_2', trainable=True)

            self.pool_2 = pool_(self.conv_2_2)

            self.conv_3_1 = conv_(self.pool_2, 256, 'conv3_1', trainable=True)
            self.conv_3_2 = conv_(self.conv_3_1, 256, 'conv3_2', trainable=True)
            self.conv_3_3 = conv_(self.conv_3_2, 256, 'conv3_3', trainable=True)

            self.pool_3 = pool_(self.conv_3_3)

            self.conv_4_1 = conv_(self.pool_3, 512, 'conv4_1', trainable=True)
            self.conv_4_2 = conv_(self.conv_4_1, 512, 'conv4_2', trainable=True)
            self.conv_4_3 = conv_(self.conv_4_2, 512, 'conv4_3', trainable=True)

            self.pool_4 = pool_(self.conv_4_3)

            self.conv_5_1 = conv_(self.pool_4, 512, 'conv5_1', trainable=True)
            self.conv_5_2 = conv_(self.conv_5_1, 512, 'conv5_2', trainable=True)
            self.conv_5_3 = conv_(self.conv_5_2, 512, 'conv5_3', trainable=True)

            self.pool_5 = pool_(self.conv_5_3)
            if self.maxout:
                self.max_5 = max_out_(self.pool_5)
                self.flattened = tf.layers.flatten(self.max_5)
            else:
                self.flattened = tf.layers.flatten(self.pool_5)

            self.fc_6 = nn.dropout(fc_(self.flattened, 4096, 'fc6'), keep_prob)
            self.fc_7 = nn.dropout(fc_(self.fc_6, 4096, 'fc7'), keep_prob)
            # self.fc_8 = nn.dropout(fc_(self.fc_7, 23, 'fc8'), keep_prob)
            self.fc_8 = fc_(self.fc_7, self.label_dim, 'fc8', relu=False)
            return self.fc_8

    def init_weights(self, sess, vgg_file):
        weights_dict = np.load(vgg_file, encoding='bytes').item()
        weights_dict = {key.decode('ascii'): value for key, value in weights_dict.items()}
        with tf.variable_scope('network', reuse=True):
            for layer in ['conv1_1', 'conv1_2',
                          'conv2_1', 'conv2_2']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W', trainable=False)
                    b = tf.get_variable('b', trainable=False)
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))
        with tf.variable_scope('network', reuse=True):
            for layer in ['conv3_1', 'conv3_2', 'conv3_3',
                          'conv4_1', 'conv4_2', 'conv4_3',
                          'conv5_1', 'conv5_2', 'conv5_3']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W')
                    b = tf.get_variable('b')
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))
