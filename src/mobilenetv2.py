''''
  Author       : Bao Jiarong
  Creation Date: 2020-06-28
  email        : bao.salirong@gmail.com
  Task         : Bottleneck & MobileNetv2
 '''

import tensorflow as tf

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,k = 32, t = 1, c = 16, n = 1, s = 1):
        super(Bottleneck, self).__init__()
        self.s = s
        self.c = c
        if self.s == 1:
            self.conv4 = tf.keras.layers.Conv2D(filters = c,kernel_size = (1,1),strides = (1,1), activation = "linear", padding = "same")
            # self.bn    = tf.keras.layers.BatchNormalization()
            c = k

        S = s
        self.convs = []
        for i in range(n):
            conv1 = tf.keras.layers.Conv2D(filters=t*k,kernel_size = (1,1),strides = (1,1), activation=tf.nn.relu6,padding = "same")
            # bn1   = tf.keras.layers.BatchNormalization()
            conv2 = tf.keras.layers.DepthwiseConv2D(  kernel_size = (3,3),strides = (S,S), activation=tf.nn.relu6,padding = "same", depth_multiplier=1)
            bn2   = tf.keras.layers.BatchNormalization()
            conv3 = tf.keras.layers.Conv2D(filters = c,kernel_size = (1,1),strides = (1,1),padding = "same")
            bn3   = tf.keras.layers.BatchNormalization()
            self.convs.extend([conv1,conv2,bn2,conv3,bn3])
            S = 1

    def call(self, inputs, training = None):
        N = 10
        x = self.convs[0](inputs)     ; #print(x.shape)
        for conv in self.convs[1:]:
            x = conv(x); #print(x.shape)

        if self.s == 1:
            x = x + inputs
            x = self.conv4(x)
            # x = self.bn(x)

        return x

#-------------------------------------------------------------------------------
class Mobilenetv2(tf.keras.Model):
    def __init__(self, classes, filters = 8, include_top = True):
        super(Mobilenetv2,self).__init__()
        self.include_top = include_top

        self.conv1 = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (3,3),strides = (2,2), activation = "relu",padding = "same")
        self.bottle_1 = Bottleneck(k = filters * 4,  t = 1, c = filters * 2,  n = 1, s = 1)
        self.bottle_2 = Bottleneck(k = filters * 2,  t = 6, c = filters * 3,  n = 2, s = 2)
        self.bottle_3 = Bottleneck(k = filters * 3,  t = 6, c = filters * 4,  n = 3, s = 2)
        self.bottle_4 = Bottleneck(k = filters * 4,  t = 6, c = filters * 8,  n = 4, s = 2)
        self.bottle_5 = Bottleneck(k = filters * 8,  t = 6, c = filters * 12, n = 3, s = 1)
        self.bottle_6 = Bottleneck(k = filters * 12, t = 6, c = filters * 20, n = 3, s = 2)
        self.bottle_7 = Bottleneck(k = filters * 20, t = 6, c = filters * 40, n = 1, s = 1)
        # self.drop     = tf.keras.layers.Dropout(0.2)
        if self.include_top == True:
            self.conv2 = tf.keras.layers.Conv2D(filters = 1280, kernel_size = (1,1),strides = (1,1), activation = "relu",padding = "same")
            # self.globalaverage = tf.keras.layers.AveragePooling2D(strides = (1,1))
            self.globalaverage = tf.keras.layers.GlobalAveragePooling2D()
            # self.out = tf.keras.layers.Conv2D(filters = classes, kernel_size = (1,1),strides = (1,1), activation = "softmax",padding = "same")
            self.out   = tf.keras.layers.Dense(units = classes, activation ="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs; #print(x.shape)
        x = self.conv1(x); #print(x.shape)
        # print("\nbottle_1 Input",x.shape)
        x = self.bottle_1(x)
        # print("\nbottle_2 Input",x.shape)
        x = self.bottle_2(x)
        # print("\nbottle_3 Input",x.shape)
        x = self.bottle_3(x)
        # print("\nbottle_4 Input",x.shape)
        x = self.bottle_4(x)
        # print("\nbottle_5 Input",x.shape)
        x = self.bottle_5(x)
        # print("\nbottle_6 Input",x.shape)
        x = self.bottle_6(x)
        # print("\nbottle_7 Input",x.shape)
        x = self.bottle_7(x)

        # Top
        if self.include_top == True:
            # print("\nconv2 Input",x.shape)
            x = self.conv2(x)
            # print("\nglobalaverage Input",x.shape)
            x = self.globalaverage(x)
            # print("\nconv3 Input",x.shape)
            x = self.out(x)
            return x

#------------------------------------------------------------------------------
def MobileNetv2(input_shape, classes, filters = 8):
    model = Mobilenetv2(classes)
    model.build(input_shape = input_shape)
    return model
