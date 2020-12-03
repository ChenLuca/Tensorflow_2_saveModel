import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.saved_model import tag_constants
import cv2 as cv
import numpy as np
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




"""     Define Data Generator      """           

dataset_path_new = "./cats_and_dogs_filtered/"

validation_dir = os.path.join(dataset_path_new, "validation")
train_dir = os.path.join(dataset_path_new, "train")

data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

"""     Define Data Generator      """       





"""     Load .h5 weight      """           

h5_file = "./1606980585.h5"

reloaded = tf.keras.models.load_model(h5_file)

# reloaded.summary()

print("type(reloaded): ", type(reloaded))

valid_loss, valid_accuracy = reloaded.evaluate_generator(valid_generator)

print("Validation accuracy after fine tuning: {}".format(valid_accuracy))


image_name = "./test_data/dog.2000.jpg"

"""Open and preprocess the image with TensorFlow:"""

tf_io_image = tf.io.read_file(image_name)
tf_io_image = tf.image.decode_image(tf_io_image)

img = cv.imshow("tf_io_image", tf_io_image.numpy())
cv.waitKey(0)

tf_io_image = tf.image.resize(tf_io_image, (128, 128))
tf_io_image = tf.expand_dims(tf_io_image, axis=0) / 255.0


np_cv_img = cv.imread(image_name)
np_cv_img = cv.cvtColor(np_cv_img, cv.COLOR_BGR2RGB) # cv2 defaults to bgr order

img = cv.imshow("np_cv_img", np_cv_img)
cv.waitKey(0)

np_cv_img = cv.resize(np_cv_img, (128, 128))
np_cv_img = np_cv_img / 255.0
np_cv_img = np_cv_img[np.newaxis, ...].astype(np.float32)
print("type of tf.image.decode_image(np_cv_img) : ", np_cv_img.dtype)

Y = reloaded.predict(tf_io_image)
print("tf_io_image Y : ", Y)

Y = reloaded.predict(np_cv_img)
print("np_cv_img Y : ", Y)


cv.destroyAllWindows()

"""     Load .h5 weight      """           




# """     Save saved_model      """   

# export_path_sm = "./checkpoints/20201103"

# #tf.saved_model.save(reloaded, export_path_sm)

# """     Save saved_model      """           




# """     Load saved_model      """       

# #reloaded_sm = tf.saved_model.load(export_path_sm)
# reload_sm_keras = tf.keras.models.load_model(export_path_sm)
# print("type(reloaded_sm): ", type(reload_sm_keras))


# EPOCHS = 10
# checkpoint_filepath = './checkpoints/20201103_checkpoint'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=False,
#     save_freq="epoch")

# reload_sm_keras.fit(train_generator,  
#                     epochs=EPOCHS, 
#                     validation_data=valid_generator,
#                     callbacks=[model_checkpoint_callback])
# valid_loss, valid_accuracy = reload_sm_keras.evaluate_generator(valid_generator)

# print("Validation accuracy after reload and fine tuning: {}".format(valid_accuracy))

# """     Load saved_model      """ 