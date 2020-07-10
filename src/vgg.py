''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-18
  email        : bao.salirong@gmail.com
  Task         : Custom layers
 '''

import tensorflow as tf

#==========================VGG based on Keras Model=============================
class Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides = (1,1), n = 1):
        super(Block, self).__init__()
        self.convs = []
        for _ in range(n):
            self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size,
                                                     strides = strides, activation  = "relu",
                                                     padding = "same"))
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

    def call(self, inputs, **kwargs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        return x

class VGG(tf.keras.Model):
    def __init__(self, classes=None, model_name = "vgg16", filters = 64, include_top = True):
        super(VGG, self).__init__()

        m = 1 if model_name == "vgg11" else 2

        if model_name == "vgg16":
            n = 3
        elif model_name == "vgg11" or model_name == "vgg13":
            n = 2
        else:
            n = 4

        self.include_top = include_top
        self.block1 = Block(filters     , kernel_size=(3,3), n = m)
        self.block2 = Block(filters << 1, kernel_size=(3,3), n = m)
        self.block3 = Block(filters << 2, kernel_size=(3,3), n = n)
        self.block4 = Block(filters << 3, kernel_size=(3,3), n = n)
        self.block5 = Block(filters << 3, kernel_size=(3,3), n = n)

        if self.include_top == True:
            self.flatten= tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units =    4096, activation="relu")
            self.dense2 = tf.keras.layers.Dense(units =    4096, activation="relu")
            self.dense3 = tf.keras.layers.Dense(units = classes, activation="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        if self.include_top == True:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)

        return x
