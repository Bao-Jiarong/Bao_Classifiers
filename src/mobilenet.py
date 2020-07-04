''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-20
  email        : bao.salirong@gmail.com
  Task         : MobileNet
 '''

import tensorflow as tf

class Block(tf.keras.layers.Layer):
    def __init__(self, strides, n):
        super(Block, self).__init__()

        self.depthconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides = strides,depth_multiplier=1,padding="same")
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.activations.relu
        self.conv1 = tf.keras.layers.Conv2D(filters = n,kernel_size = (1,1),strides = (1,1), activation  = "relu",padding = "same")
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.activations.relu

    def call(self, inputs, training = None):
        x = inputs
        x = self.depthconv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

#-------------------------------------------------------------------------------
class Mobilenet(tf.keras.Model):
    def __init__(self, classes, filters = 64, include_top = True):
        super(Mobilenet,self).__init__()
        self.include_top = include_top

        self.conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3),strides = (2,2), activation = "relu",padding = "same")
        self.block1 = Block(1, 64)
        self.block2 = Block(2, filters * 2)
        self.block3 = Block(1, filters * 2)
        self.block4 = Block(2, filters * 4)
        self.block5 = Block(1, filters * 4)
        self.block6 = Block(2, filters * 8)

        self.block7 = Block(1, filters * 8)
        self.block8 = Block(2, filters * 16)
        self.block9 = Block(2, filters * 16)

        if self.include_top == True:
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.fc   = tf.keras.layers.Dense(units = classes, activation ="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        for i in range(5):
            x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        # Top
        if self.include_top == True:
            x = self.pool(x)
            x = self.fc(x)
            return x

#------------------------------------------------------------------------------
def MobileNet(input_shape, classes, filters = 64):
    model = Mobilenet(classes,filters)
    model.build(input_shape = input_shape)
    return model
