''''
  Author       : Bao Jiarong
  Creation Date: 2020-07-07
  email        : bao.salirong@gmail.com
  Task         : ResNet
 '''

import tensorflow as tf
import sys

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, is_begaining = False):
        super(Block, self).__init__()
        self.is_begaining = is_begaining
        self.convs = []
        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3),
                                                 strides = strides, activation  = "relu",
                                                 padding = "same"))

        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3),
                                                 strides = (1,1), activation  = "linear",
                                                 padding = "same"))

        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1),strides = strides, activation = "linear",padding = "same")

    def call(self, inputs, **kwargs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        if self.is_begaining == True:
            inputs = self.conv2(inputs)
        x = x + inputs
        x = tf.keras.activations.relu(x)
        return x

class bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, is_begaining = False):
        super(bottleneck, self).__init__()
        self.is_begaining = is_begaining
        self.convs = []
        f = filters * 4
        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1),
                                                 strides = strides, activation  = "relu",
                                                 padding = "same"))

        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3),
                                                 strides = (1,1), activation  = "relu",
                                                 padding = "same"))

        self.convs.append(tf.keras.layers.Conv2D(filters = f, kernel_size = (1,1),
                                                 strides = (1,1), activation  = "linear",
                                                 padding = "same"))

        self.conv2 = tf.keras.layers.Conv2D(filters = f, kernel_size = (1,1),strides = strides, activation = "linear",padding = "same")

    def call(self, inputs, **kwargs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        if self.is_begaining == True:
            inputs = self.conv2(inputs)
        x = x + inputs
        x = tf.keras.activations.relu(x)
        return x

#-------------------------------------------------------------------------------
class Resnet_1(tf.keras.Model):
    def __init__(self, classes, model_name = "resnet18", filters = 64, include_top = True):
        super(Resnet_1,self).__init__()
        self.model_name = model_name
        self.include_top = include_top
        # Backbone
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (7,7),strides = (2,2), activation = "relu",padding = "same")
        self.pool1  = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))

        self.block1_1 = Block(filters)
        self.block1_2 = Block(filters)
        if self.model_name == "resnet34":
            self.block1_3 = Block(filters)

        self.block2_1 = Block(filters << 1, 2, True)
        self.block2_2 = Block(filters << 1)
        if self.model_name == "resnet34":
            self.block2_3 = Block(filters << 1)
            self.block2_4 = Block(filters << 1)

        self.block3_1 = Block(filters << 2, 2, True)
        self.block3_2 = Block(filters << 2)
        if self.model_name == "resnet34":
            self.block3_3 = Block(filters << 2)
            self.block3_4 = Block(filters << 2)
            self.block3_5 = Block(filters << 2)
            self.block3_6 = Block(filters << 2)

        self.block4_1 = Block(filters << 3, 2, True)
        self.block4_2 = Block(filters << 3)
        if self.model_name == "resnet34":
            self.block4_3 = Block(filters << 3)

        # Top
        if self.include_top == True:
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.fc   = tf.keras.layers.Dense(units = classes, activation ="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        if self.model_name == "resnet34":
            x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        if self.model_name == "resnet34":
            x = self.block2_3(x)
            x = self.block2_4(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        if self.model_name == "resnet34":
            x = self.block3_3(x)
            x = self.block3_4(x)
            x = self.block3_5(x)
            x = self.block3_6(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        if self.model_name == "resnet34":
            x = self.block4_3(x)

        # Top
        if self.include_top == True:
            x = self.pool(x)
            x = self.fc(x)
        return x

class Resnet_2(tf.keras.Model):
    def __init__(self, classes, model_name = "resnet50", filters = 64, include_top = True):
        super(Resnet_2,self).__init__()
        self.model_name = model_name
        self.include_top = include_top
        # Backbone
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (7,7),strides = (2,2), activation = "relu",padding = "same")
        self.pool1  = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))

        self.block1_1 = bottleneck(filters, 1, True)
        self.block1_2 = bottleneck(filters)
        self.block1_3 = bottleneck(filters)

        self.block2_1 = bottleneck(filters << 1, 2, True)
        self.block2_2 = bottleneck(filters << 1)
        self.block2_3 = bottleneck(filters << 1)
        self.block2_4 = bottleneck(filters << 1)
        if self.model_name == "resnet152":
            self.block2_5 = bottleneck(filters << 1)
            self.block2_6 = bottleneck(filters << 1)
            self.block2_7 = bottleneck(filters << 1)
            self.block2_8 = bottleneck(filters << 1)

        self.block3_1  = bottleneck(filters << 2, 2, True)
        self.block3_2  = bottleneck(filters << 2)
        self.block3_3  = bottleneck(filters << 2)
        self.block3_4  = bottleneck(filters << 2)
        self.block3_5  = bottleneck(filters << 2)
        self.block3_6  = bottleneck(filters << 2)
        if self.model_name != "resnet50":
            self.block3_7  = bottleneck(filters << 2)
            self.block3_8  = bottleneck(filters << 2)
            self.block3_9  = bottleneck(filters << 2)
            self.block3_10 = bottleneck(filters << 2)
            self.block3_11 = bottleneck(filters << 2)
            self.block3_12 = bottleneck(filters << 2)
            self.block3_13 = bottleneck(filters << 2)
            self.block3_14 = bottleneck(filters << 2)
            self.block3_15 = bottleneck(filters << 2)
            self.block3_16 = bottleneck(filters << 2)
            self.block3_17 = bottleneck(filters << 2)
            self.block3_18 = bottleneck(filters << 2)
            self.block3_19 = bottleneck(filters << 2)
            self.block3_20 = bottleneck(filters << 2)
            self.block3_21 = bottleneck(filters << 2)
            self.block3_22 = bottleneck(filters << 2)
            self.block3_23 = bottleneck(filters << 2)
            if self.model_name == "resnet152":
                self.block3_24 = bottleneck(filters << 2)
                self.block3_25 = bottleneck(filters << 2)
                self.block3_26 = bottleneck(filters << 2)
                self.block3_27 = bottleneck(filters << 2)
                self.block3_28 = bottleneck(filters << 2)
                self.block3_29 = bottleneck(filters << 2)
                self.block3_30 = bottleneck(filters << 2)
                self.block3_31 = bottleneck(filters << 2)
                self.block3_32 = bottleneck(filters << 2)
                self.block3_33 = bottleneck(filters << 2)
                self.block3_34 = bottleneck(filters << 2)
                self.block3_35 = bottleneck(filters << 2)
                self.block3_36 = bottleneck(filters << 2)

        self.block4_1 = bottleneck(filters << 3, 2, True)
        self.block4_2 = bottleneck(filters << 3)
        self.block4_3 = bottleneck(filters << 3)

        # Top
        if self.include_top == True:
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.fc   = tf.keras.layers.Dense(units = classes, activation ="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)
        if self.model_name == "resnet152":
            x = self.block2_5(x)
            x = self.block2_6(x)
            x = self.block2_7(x)
            x = self.block2_8(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        x = self.block3_6(x)
        if self.model_name != "resnet50":
            x = self.block3_7(x)
            x = self.block3_8(x)
            x = self.block3_9(x)
            x = self.block3_10(x)
            x = self.block3_11(x)
            x = self.block3_12(x)
            x = self.block3_13(x)
            x = self.block3_14(x)
            x = self.block3_15(x)
            x = self.block3_16(x)
            x = self.block3_17(x)
            x = self.block3_18(x)
            x = self.block3_19(x)
            x = self.block3_20(x)
            x = self.block3_21(x)
            x = self.block3_22(x)
            x = self.block3_23(x)
            if self.model_name == "resnet152":
                x = self.block3_24(x)
                x = self.block3_25(x)
                x = self.block3_26(x)
                x = self.block3_27(x)
                x = self.block3_28(x)
                x = self.block3_29(x)
                x = self.block3_30(x)
                x = self.block3_31(x)
                x = self.block3_32(x)
                x = self.block3_33(x)
                x = self.block3_34(x)
                x = self.block3_35(x)
                x = self.block3_36(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)

        # Top
        if self.include_top == True:
            x = self.pool(x)
            x = self.fc(x)
        return x
