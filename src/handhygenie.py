import cv2
import mediapipe as mp
import numpy as np
from cnn_model import load_model 
from utils import predict_hygiene

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Load the pre-trained model for hygiene level prediction
model_path = 'cnn_model.h5'  # Update this path to your actual model file
try:
    model = load_model(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    # Debugging: Print if hands are detected
    if result.multi_hand_landmarks:
        print("Hands detected")
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the hand region
            h, w, c = frame.shape
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
            hand_image = frame[y_min:y_max, x_min:x_max]
            
            # Predict hygiene level
            if hand_image.size > 0:  # Ensure the hand image is not empty
                hygiene_level = predict_hygiene(model, hand_image)
                
                # Display hygiene level on the frame
                cv2.putText(frame, f'Hygiene Level: {hygiene_level:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("No hands detected")
    
    # Display the frame
    cv2.imshow('Hand Hygiene Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
