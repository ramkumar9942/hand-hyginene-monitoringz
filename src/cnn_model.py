import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Adjust for multi-class if needed
    return model

def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def train_model(model, train_data, train_labels, validation_data, validation_labels, epochs=10, batch_size=32):
    history = model.fit(train_data, train_labels, 
                        validation_data=(validation_data, validation_labels),
                        epochs=epochs, 
                        batch_size=batch_size)
    return history

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    return models.load_model(filepath)