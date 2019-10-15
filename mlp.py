from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten,Activation,ZeroPadding2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from sklearn.preprocessing import LabelBinarizer

import numpy as np
import tensorflow as tf
import random

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)
# path to dataset
dataset_path = "gdrive/My Drive/University VUW/COMP309/Project/COMP309-Project/Train_data"
image_size_tuple = (300,300)
image_size =300
num_classes=3
batch_size=16

def load_data():
    datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

    train = datagen.flow_from_directory(dataset_path, target_size=image_size_tuple,
                                                      classes=['cherry', 'strawberry', 'tomato'], batch_size=batch_size ,
                                                      class_mode='categorical', subset='training')

    test = datagen.flow_from_directory(dataset_path, target_size=image_size_tuple,
                                                      classes=['cherry', 'strawberry', 'tomato'], batch_size=batch_size ,
                                                      class_mode='categorical', subset='validation')

    return train, test


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()

    # First layer block
    model.add(Dense(32, input_shape=(image_size, image_size, 3), activation="relu"))

    # Second layer block
    model.add(Dense(32, activation="relu"))

    # Third layer block
    model.add(Dense(32, activation="relu"))

    # Dropout layer
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Classification layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def train_model(model,train,test):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here
    model.fit_generator(train,steps_per_epoch=2000,validation_data=test,epochs=15)
    return model

def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    # model.save("model/model.h5")
    print("Model Saved Successfully.")
    model.save_weights('gdrive/My Drive/University VUW/COMP309/Project/model/mlp.h5')

x,y=load_data()
model = construct_model()
trained_model = train_model(model,x,y)