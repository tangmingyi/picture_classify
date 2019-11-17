# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SE_ResNet_Xt():
    def __init__(self, input,layers,class_dim):
        self.layers = layers
        self.net(input=input,class_dim=class_dim)

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act=tf.nn.relu,
                name='conv1', )
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=3, strides=3, padding='same')

        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act=tf.nn.relu,
                name="conv1", )
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=3, strides=2, padding='same')

        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act=tf.nn.relu,
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act=tf.nn.relu,
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act=tf.nn.relu,
                name='conv3')
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=3, strides=2, padding='same')

        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                    name=str(n) + '_' + str(i + 1))
        pool = self.global_avg_pooling_2d(conv,False)
        drop = tf.layers.dropout(inputs=pool, rate=0.5)
        stdv = tf.divide(tf.constant(1, tf.float32), tf.sqrt(tf.cast(drop.shape[1], tf.float32)))
        self.cls = tf.layers.dense(inputs=drop, units=class_dim,
                              kernel_initializer=tf.initializers.random_uniform(minval=-stdv, maxval=stdv))

    def get_cls_layer(self):
        return self.cls

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[-1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride, name='conv' + name + '_prj')
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         cardinality,
                         reduction_ratio,
                         name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act=tf.nn.relu,
            name='conv' + name + '_x1')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act=tf.nn.relu,
            name='conv' + name + '_x2')
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            name='fc' + name)

        short = self.shortcut(input, num_filters * 2, stride, name=name)

        return tf.nn.relu(tf.add(short, scale))

    def global_avg_pooling_2d(self, input, keep_dim,name=None):
        return tf.reduce_mean(input_tensor=input, axis=[1, 2], name=name, keep_dims=keep_dim)

    def groups_conv2d(self, input, groups, num_filters, kernel_size, strides, padding):
        """
        分组卷积。
        :param input:
        :param groups:
        :param num_filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :return:
        """
        input_dim = input.shape[-1].value
        input.get_shape()
        if input_dim % groups != 0:
            raise ValueError('The number of input channels is not divisible '
                             'by the number of channel group. %d %% %d = %d' %
                             (input_dim, groups, input_dim % groups))
        if num_filters % groups != 0:
            raise ValueError('The number of output channels is not divisible '
                             'by the number of channel group. %d %% %d = %d' %
                             (num_filters, groups, num_filters % groups))
        if (groups == 1):
            return tf.layers.conv2d(inputs=input,filters=num_filters, kernel_size=kernel_size, strides=strides,
                                    padding='same')
        else:
            kernel = tf.Variable(
                initial_value=tf.random_normal(shape=[kernel_size, kernel_size, input_dim//groups, num_filters]))
            input_slices = tf.split(value=input, num_or_size_splits=groups, axis=-1)
            kernel_slices = tf.split(value=kernel, num_or_size_splits=groups, axis=-1)
            out_slices = [
                tf.nn.conv2d(input_slice, kernel_slice, padding=padding.upper(), strides=[1, strides, strides, 1]) for
                input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            return tf.concat(values=out_slices, axis=-1)

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = self.groups_conv2d(input=input, groups=groups, num_filters=num_filters, kernel_size=filter_size,
                                  strides=stride, padding='same')
        bn_name = name + "_bn"
        if (act != None):
            return act(self.batch_normalization(conv, bn_name))
        return self.batch_normalization(conv, bn_name)

    def batch_normalization(self, inputs, name):
        axis = list(range(len(inputs.get_shape()) - 1))
        mean, var = tf.nn.moments(x=inputs, axes=axis)
        return tf.nn.batch_normalization(x=inputs, mean=mean,
                                         variance=var,
                                         variance_epsilon=0.001,
                                         name=name,
                                         scale=tf.Variable(initial_value=tf.ones(shape=mean.get_shape())),
                                         offset=tf.Variable(initial_value=tf.zeros(shape=mean.get_shape()))
                                         )

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = self.global_avg_pooling_2d(input,True)
        stdv = tf.divide(tf.constant(1, tf.float32), tf.sqrt(tf.cast(pool.shape[1], tf.float32)))
        squeeze = tf.layers.dense(inputs=pool, units=num_channels // reduction_ratio,
                                  kernel_initializer=tf.initializers.random_uniform(minval=-stdv, maxval=stdv))

        stdv = tf.divide(tf.constant(1, tf.float32), tf.sqrt(tf.cast(squeeze.shape[1], tf.float32)))
        excitation = tf.layers.dense(inputs=squeeze, units=num_channels,
                                     kernel_initializer=tf.initializers.random_uniform(minval=-stdv, maxval=stdv),
                                     activation=tf.nn.sigmoid)
        scale = tf.multiply(x=input, y=excitation)
        return scale
