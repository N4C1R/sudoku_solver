import tensorflow as tf
from keras import layers, models
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def create_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(9, 9, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense (fully connected) layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def load_trained_model(model_path='models/sudoku_solver_model.h5'):
    # Load the pre-trained model
    loaded_model = load_model(model_path)
    return loaded_model


def preprocess_image(image_path):
    # Load the image and preprocess it for the model
    img = load_img(image_path, color_mode='grayscale', target_size=(9, 9))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array
