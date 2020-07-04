''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-20
  email        : bao.salirong@gmail.com
  Task         : LeNet based on Keras Model
 '''

import tensorflow as tf
#==========================LeNet based on Keras Model===========================
class Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides = (1,1), n = 1, activation = "tanh"):
        super(Block, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size,
                                                     strides = strides, activation  = activation,
                                                     padding = "valid")
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv(x)
        x = self.pool(x)
        return x

class LeNet(tf.keras.Model):
    def __init__(self, classes=None, include_top = True):
        super(LeNet, self).__init__()
        self.include_top = include_top

        self.block1 = Block(6  , kernel_size=(5,5), n = 1)
        self.block2 = Block(16 , kernel_size=(5,5), n = 1)
        self.block3 = tf.keras.layers.Conv2D(filters = 120, kernel_size = (5,5),
                                             strides = (1,1), activation  = "tanh",
                                             padding = "valid")
        if self.include_top == True:
            self.flatten= tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units = 84, activation = "tanh")
            self.dense2 = tf.keras.layers.Dense(units = classes, activation="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Top
        if self.include_top == True:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return x

#------------------------------------------------------------------------------
def LeNet_5(input_shape, classes):
    model = LeNet(classes)
    model.build(input_shape = input_shape)
    return model
