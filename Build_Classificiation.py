# -*- coding: utf-8 -*-
import argparse

import os
import zipfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import datetime 

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

solve_cudnn_error()

t = time.time()

day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]
current_time = day + "_" + time

parser = argparse.ArgumentParser()

parser.add_argument('--Dataset_dir', type=str, default='./Datasets', help='Path to dataset')
parser.add_argument('--Num_Class', type=int, default=2, help='Number of class')
parser.add_argument('--Batch_size', type=int, default=64, help='Number of batch size')
parser.add_argument('--Pre_Train_Epochs', type=int, default=5, help='Number of pre train epochs')
parser.add_argument('--Fine_Tuning_Epochs', type=int, default=10, help='Number of fine tuning epochs')
parser.add_argument('--Save_model_dir', type=str, default='./Save_model', help='Path to save model')

FLAGS = parser.parse_args()

"""### Seting up Hyper Params"""

NUM_CLASS = FLAGS.Num_Class
BATCH_SIZE = FLAGS.Batch_size
PRE_TRAIN_EPOCHS = FLAGS.Pre_Train_Epochs
FINE_TUNING_EPOCHS =FLAGS.Fine_Tuning_Epochs
TRAIN_DIR = FLAGS.Dataset_dir
SAVE_MODEL_DIR = FLAGS.Save_model_dir
IMG_SHAPE = (128, 128, 3)

export_path_keras = "/{}.h5".format(TRAIN_DIR.split('/')[-2] + "__" + current_time)

print("\nModel name:{}\n".format(export_path_keras))

print("Train hyper params:\n NUM_CLASS={}\n BATCH_SIZE={}\n DATASET={}\n PRE_TRAIN_EPOCHS={}\n FINE_TUNING_EPOCHS={}\n".format(NUM_CLASS,
                                                                                                                                    BATCH_SIZE,
                                                                                                                                    TRAIN_DIR,
                                                                                                                                    PRE_TRAIN_EPOCHS,
                                                                                                                                    FINE_TUNING_EPOCHS))


"""## Building ImageDataGenerator"""

data_gen_train = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range = 180,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    brightness_range = (0.8, 1.2),
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    vertical_flip = True, 
                                    validation_split=0.2)

train_generator = data_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=TRAIN_DIR,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE[0], IMG_SHAPE[1]), 
                                                subset="training",
                                                class_mode='categorical')

valid_generator = data_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=TRAIN_DIR,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE[0], IMG_SHAPE[1]), 
                                                subset="validation",
                                                class_mode='categorical')
"""## Building the model

### Loading the pre-trained model (MobileNetV2)

"""

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

# base_model.summary()

"""### Freezing the base model"""

base_model.trainable = False

"""### Defining the custom head for our network"""

base_model.output

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

prediction_layer = tf.keras.layers.Dense(units=NUM_CLASS, activation='softmax')(global_average_layer)

"""### Defining the model"""

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

# model.summary()

"""### Compiling the model"""

model.compile(optimizer='adam',
            loss="categorical_crossentropy", 
            metrics=["accuracy"])

"""### Creating Data Generators

Resizing images

    Big pre-trained architecture support only certain input sizes.

For example: MobileNet (architecture that we use) supports: (96, 96), (128, 128), (160, 160), (192, 192), (224, 224).
"""

"""### Training the model"""
print("Pre-training...")
model.fit(train_generator, 
            epochs=PRE_TRAIN_EPOCHS,
            validation_data=valid_generator)

"""## Fine tuning


There are a few pointers:

- DO NOT use Fine tuning on the whole network; only a few top layers are enough. In most cases, they are more specialized. The goal of the Fine-tuning is to adopt that specific part of the network for our custom (new) dataset.
- Start with the fine tunning AFTER you have finished with transfer learning step. If we try to perform Fine tuning immediately, gradients will be much different between our custom head layer and a few unfrozen layers from the base model.

### Un-freeze a few top layers from the model
"""

base_model.trainable = True

# print("Number of layersin the base model: {}".format(len(base_model.layers)))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

"""### Compiling the model for fine-tuning"""

model.compile(optimizer='adam',
            loss="categorical_crossentropy", 
            metrics=["accuracy"])

"""### Fine tuning"""

print("Fine tuning...")
model.fit(train_generator, 
            epochs=FINE_TUNING_EPOCHS,
            validation_data=valid_generator)

"""### Save model"""

model.save(SAVE_MODEL_DIR + export_path_keras)