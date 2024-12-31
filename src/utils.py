import cv2
import numpy as np

def predict_hygiene(model, hand_image):
    img = cv2.resize(hand_image, (128, 128))  # Resize to match the input shape of the model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction[0][0]  # Adjust based on your model's output