''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-18
  email        : bao.salirong@gmail.com
  Task         : Custom layers
 '''

import tensorflow as tf

class fireblock(tf.keras.layers.Layer):
    def __init__(self, n, m):
        super(fireblock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = n,     kernel_size = (1,1),
                                             strides = (1,1), activation  = "relu",
                                             padding = "same")
        self.conv2 = tf.keras.layers.Conv2D(filters = m,     kernel_size = (1,1),
                                             strides = (1,1), activation  = "relu",
                                             padding = "same")
        self.conv3 = tf.keras.layers.Conv2D(filters = m,     kernel_size = (3,3),
                                             strides = (1,1), activation  = "relu",
                                             padding = "same")

    def call(self, inputs, training = None):
        x = inputs
        x = self.conv1(x)
        L = self.conv2(x)
        R = self.conv3(L)
        x = tf.keras.layers.Concatenate(axis = 3)([L,R])
        return x

#-------------------------------------------------------------------------------
class Squeezenetv3(tf.keras.Model):
    def __init__(self, classes, filters = 16, include_top = True):
        super(Squeezenetv3,self).__init__()
        self.include_top = include_top

        self.conve1 = tf.keras.layers.Conv2D(filters = filters * 8 , kernel_size = (1,1),strides = (1,1), activation = "relu",padding = "valid")
        self.conve2 = tf.keras.layers.Conv2D(filters = filters * 16, kernel_size = (1,1),strides = (1,1), activation = "relu",padding = "valid")
        self.conve3 = tf.keras.layers.Conv2D(filters = filters * 24, kernel_size = (1,1),strides = (1,1), activation = "relu",padding = "valid")
        self.conve4 = tf.keras.layers.Conv2D(filters = filters * 32, kernel_size = (1,1),strides = (1,1), activation = "relu",padding = "valid")

        self.conv1 = tf.keras.layers.Conv2D(filters = filters * 6, kernel_size = (7,7),strides = (2,2), activation = "relu",padding = "valid")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.fire2 = fireblock(filters,    filters * 4)
        self.fire3 = fireblock(filters,    filters * 4)
        self.fire4 = fireblock(filters * 3,filters * 8)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.fire5 = fireblock(filters * 3, filters * 8)
        self.fire6 = fireblock(filters * 4, filters * 12)
        self.fire7 = fireblock(filters * 4, filters * 12)
        self.fire8 = fireblock(filters * 4, filters * 16)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.fire9 = fireblock(filters * 4, filters * 16)
        self.conv10 = tf.keras.layers.Conv2D(filters = classes, kernel_size = (1,1),strides = (1,1),padding = "same")

        if self.include_top == True:
            self.globalaverage = tf.keras.layers.GlobalAveragePooling2D()
            self.softmax = tf.keras.activations.softmax

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)
        x11 = self.conve1(x)
        x2= self.fire2(x)
        y = x11 + x2
        x3= self.fire3(y)
        x31 = self.conve2(x3)
        y = x2 + x3
        x = self.fire4(y)
        y = x31 + x
        x4 = self.pool2(y)
        x5 = self.fire5(x4)
        x51 = self.conve3(x5)
        y = x4 + x5
        x6 = self.fire6(y)
        y = x51 + x6
        x7 = self.fire7(y)
        x71 = self.conve4(x7)
        y = x6 + x7
        x = self.fire8(y)
        y = x71 + x
        x8 = self.pool3(y)
        x9 = self.fire9(x8)
        y = x8 + x9
        x = self.conv10(y)

        # Top
        if self.include_top == True:
            x = self.globalaverage(x)
            x = self.softmax(x)
            return x
