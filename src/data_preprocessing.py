import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

def preprocess_images(images, target_size=(128, 128)):
    processed_images = []
    for img in images:
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize the image
        processed_images.append(img)
    return np.array(processed_images)

def augment_images(images, labels, batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(np.array(images), np.array(labels), batch_size=batch_size)