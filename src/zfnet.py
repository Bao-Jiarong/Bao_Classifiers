''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-20
  email        : bao.salirong@gmail.com
  Task         : ZFNet based on Keras Model
 '''

import tensorflow as tf
#==========================ZFNet based on Keras Model===========================
class Block(tf.keras.layers.Layer):
    def __init__(self, n, kernel_size, strides = (1,1), padding='same'):
        super(Block, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters = n, kernel_size = kernel_size,
                                                 strides = strides, activation  = "relu",
                                                 padding = padding)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv(x)
        x = self.pool(x)
        return x

class ZFnet(tf.keras.Model):
    def __init__(self, classes = None, filters = 32, include_top = True):
        super(ZFnet, self).__init__()
        self.include_top = include_top

        self.block1 = Block(n = filters * 3 , kernel_size=(7,7), strides=(2,2), padding='valid')
        self.block2 = Block(n = filters * 8 , kernel_size=(5,5), strides=(2,2))
        self.block3 = tf.keras.layers.Conv2D(filters = filters * 16,kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu")
        self.block4 = tf.keras.layers.Conv2D(filters = filters * 32,kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu")
        self.block5 = Block(n = filters * 16 , kernel_size=(3,3))

        if self.include_top == True:
            self.flatten= tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(units = filters * 288, activation = "relu")
            self.dense2 = tf.keras.layers.Dense(units = filters * 128, activation = "relu")
            self.dense3 = tf.keras.layers.Dense(units = filters * 128, activation = "relu")
            self.dense4 = tf.keras.layers.Dense(units = classes, activation="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Top
        if self.include_top == True:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.dense4(x)
            return x

#------------------------------------------------------------------------------
def ZFNet(input_shape, classes, filters):
    model = ZFnet(classes, filters = filters)
    model.build(input_shape = input_shape)
    return model
