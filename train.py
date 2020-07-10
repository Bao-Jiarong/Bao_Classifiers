'''
  Author       : Bao Jiarong
  Creation Date: 2020-07-09
  email        : bao.salirong@gmail.com
  Task         : Classifier
  Dataset      : MNIST Digits (0,1,...,9)
'''

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import cv2
import loader
import src.backbone

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 224 >> 2
height     = 224 >> 2
channel    = 3
n_outputs  = 10

backbone_names = ["VGG11","VGG13","VGG16","VGG19","alexnet","lenet","zfnet",
                  "resnet18","resnet34","resnet50","resnet101","resnet152",
                  "squeezenet","squeezenetv2","squeezenetv3",
                  "mobilenet","mobilenetv2"]
backbone_name = backbone_names[11]
model_name = "models/"+backbone_name+"/digists"
data_path  = "../data_img/MNIST/train/"

# Step 0: Global Parameters
epochs     = 2
lr_rate    = 0.0001
batch_size = 32

# Step 1: Create Model
model = src.backbone.Backbone((None, height, width, channel),
                              classes = n_outputs,
                              filters = 6,
                              include_top = True,
                              backbone_name = backbone_name)

# Step 2: Define Metrics
print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,True)

    # Step 4: Training
    #model.load_weights(model_name)

    # Define The Optimizer
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)

    # Define The Loss
    my_loss  = tf.keras.losses.SparseCategoricalCrossentropy()

    # Define The Metrics
    tr_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_loss')
    tr_accu = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accu')

    va_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='valid_loss')
    va_accu = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accu')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )
        tr_accu(Y_true, Y_pred)

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)

        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)
        va_accu(Y_true, Y_pred)

    #---------------------
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            images, labels = loader.get_batch_light(X_train, Y_train, batch_size, width, height)
            train_step(images,labels)

            print(epoch,"/",epochs,step,steps,
                  "loss:",tr_loss.result().numpy(),"accuracy:",tr_accu.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            images, labels = loader.get_batch_light(X_valid, Y_valid, batch_size, width, height)
            valid_step(images, labels)

        print(epoch,"/",epochs,step,steps,
              "loss:",tr_loss.result().numpy(),"accuracy:",tr_accu.result().numpy(),
              "val_loss:",va_loss.result().numpy(),"val_accuracy:",va_accu.result().numpy())
        # print("val_loss:",va_loss.result().numpy(),"val_accuracy:",va_accu.result().numpy())
        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')


elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    print(np.argmax(preds[0]))
    print(preds[0])
