
import time
import os
import cv2
import mediapipe as mp
import numpy as np

wCam, hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

with mp_hand.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, img = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = hands.process(img)


        #draw hand annotations
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style())

        cv2.imshow("Hand", cv2.flip(img, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
  
