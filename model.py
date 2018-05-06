import tensorflow as tf
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from collections import OrderedDict


class TransferModel(object):
    def __init__(self):
        self.init_weight = np.load('./vgg19.npy', encoding='latin1').item()
        # ######################################
        self.content_layers = ['conv4_2']
        self.content_weights = [1.0]
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.content_style_loss_ratio = 1e-4
        # attention: the weights were trained in the data format bgr. so may be we should fellow this
        # but here, i do not use bgr. if the result is not good, you can change it to rgb
        self.mean_pixel = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # rgb
        self.content = np.expand_dims(io.imread('./images/content/build.jpeg') - self.mean_pixel, axis=0)
        self.style = np.expand_dims(io.imread('./images/style/rain_princess.jpg') - self.mean_pixel, axis=0)

        b, h, w, c = self.content.shape
        self.rlt = tf.Variable((np.random.randn(b, h, w, c).astype(np.float32)),
                               trainable=True)
        # self.rlt = tf.Variable(self.content.copy(), trainable=True)
        # self.rlt = tf.Variable(self.style.copy(), trainable=True)

        assert len(self.style_layers) == len(self.style_weights)
        assert len(self.content_layers) == len(self.content_weights)

    def build_model(self):
        # content
        self.content_tensors_dict = self.vgg19_forward(self.content, store_objects=self.content_layers, scope='content')
        # style
        self.style_tensors_dict = self.vgg19_forward(self.style, store_objects=self.style_layers, scope='style')
        for key in self.style_tensors_dict.keys():
            self.style_tensors_dict[key] = self.gram_matrix(self.style_tensors_dict[key])

        # pre-compute
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            for key in self.content_tensors_dict.keys():
                self.content_tensors_dict[key] = sess.run(self.content_tensors_dict[key],)
            for key in self.style_tensors_dict.keys():
                self.style_tensors_dict[key] = sess.run(self.style_tensors_dict[key],)
        # rlt
        self.rlt_tensor_dict = self.vgg19_forward(self.rlt, set(self.style_layers+self.content_layers), scope='result')

    def calc_loss(self):
        # content_loss
        content_loss = 0
        for i, key in enumerate(self.content_tensors_dict.keys()):
            tp = tf.pow(x=self.rlt_tensor_dict[key]-self.content_tensors_dict[key], y=2)
            content_loss += self.content_weights[i] * tf.reduce_sum(tp) / 2

        # style loss
        style_loss = 0
        for i, key in enumerate(self.style_layers):
            _, h, w, c = self.rlt_tensor_dict[key].get_shape()
            n = h.value * w.value
            c = c.value
            gram = self.gram_matrix(self.rlt_tensor_dict[key])
            tp = tf.pow(self.style_tensors_dict[key]-gram, 2)
            style_loss += (1.0 / tf.constant(4.0*n*n*c*c, dtype=tf.float32)) * self.style_weights[i] * tf.reduce_sum(tp)

        loss = self.content_style_loss_ratio * content_loss + style_loss

        return loss, content_loss, style_loss

    def get_result_img(self, sess):
        return np.squeeze(np.clip(sess.run(self.rlt)+self.mean_pixel, 0.0, 255).astype(np.uint8))
    # ######################################################################################
    @staticmethod
    def gram_matrix(tensor):
        flat = tf.reshape(tensor, shape=(-1, tensor.shape[-1]))
        gram = tf.matmul(tf.transpose(flat), flat)
        return gram

    # ######################################################################################
    def vgg19_forward(self, x, store_objects, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.conv1_1 = self.conv_layer(x, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            print(self.conv4_2)
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        store = OrderedDict()
        for name in store_objects:
            store[name] = tf.get_default_graph().get_tensor_by_name(name=scope+'/'+name+':0')
        return store

    def conv_layer(self, x, name):
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #     filter = tf.get_variable(name='kernel', shape=self.init_weight[name][0].shape,
        #                              initializer=tf.constant_initializer(self.init_weight[name][0]), trainable=False)
        #     bias = tf.get_variable(name='bias', shape=self.init_weight[name][1].shape,
        #                            initializer=tf.constant_initializer(self.init_weight[name][1]), trainable=False)

        conv_filter = tf.constant(self.init_weight[name][0], dtype=tf.float32, name='kernel')
        conv_bias = tf.constant(self.init_weight[name][1], dtype=tf.float32, name='bias')

        conv = tf.nn.conv2d(input=x, filter=conv_filter, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_bias)
        return tf.nn.relu(bias, name=name)

    @staticmethod
    def max_pool(x, name):
        return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same', name=name)


if __name__ == '__main__':

    path = '/Users/whyguu/Desktop/build.jpeg'
    # path = './images/style/rain_princess.jpg'
    image = io.imread(path)
    w, h, _ = image.shape
    sp = np.array([w, h]) / np.max([w, h]) * 512
    sp = sp.astype(np.int)
    print(sp)
    image = resize(image, output_shape=sp)
    io.imsave('./images/content/build.jpg', image)
