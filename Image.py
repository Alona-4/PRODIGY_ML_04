import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = tf.keras.models.load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Read the input image
frame = cv2.imread("J.jpg")

# Get image dimensions
height, width, _ = frame.shape  # Unpack all three values

# Convert the image to RGB
framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Get hand landmark prediction
result = hands.process(framergb)

className = ''

if result.multi_hand_landmarks:
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * width)
            lmy = int(lm.y * height)
            landmarks.append([lmx, lmy])

        # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Predict gesture
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]

print(className)
