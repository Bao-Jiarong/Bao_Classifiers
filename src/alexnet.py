'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-20
  email        : bao.salirong@gmail.com
  Task         : AlexNet based on Keras Model
'''

import tensorflow as tf
#==========================AlexNet based on Keras Model=========================
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

class Alexnet(tf.keras.Model):
    def __init__(self, classes = None, filters = 32, include_top = True):
        super(Alexnet, self).__init__()
        self.include_top = include_top

        self.block1 = Block(n = filters * 3 , kernel_size=(11,11), strides=(4,4), padding='valid')
        self.block2 = Block(n = filters * 8 , kernel_size=(5,5), strides=(1,1))
        self.block3 = tf.keras.layers.Conv2D(filters = filters * 12,kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu")
        self.block4 = tf.keras.layers.Conv2D(filters = filters * 12,kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu")
        self.block5 = Block(n = filters * 8 , kernel_size=(3,3))

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

        if self.include_top == True:
            # Top
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.dense4(x)
        return x


#------------------------------------------------------------------------------
def AlexNet(input_shape, classes, filters):
    model = Alexnet(classes, filters = filters, include_top = True)
    model.build(input_shape = input_shape)
    return model
