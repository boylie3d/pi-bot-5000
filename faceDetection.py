import time

import cv2
import matplotlib.pyplot as plt
from fer import FER

faceCascade = cv2.CascadeClassifier('c:/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

lastTime = time.time()
emo_detector = FER(mtcnn=True)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        dominant_emotion, emotion_score = emo_detector.top_emotion(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if dominant_emotion:
                cv2.putText(frame, 
                            "{}: {}".format(dominant_emotion, emotion_score), 
                            (x, y), 
                            font,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)
            
    thisTime = time.time()
    delta = thisTime - lastTime
    framerate = int(1 / delta)  
    lastTime = thisTime
    
    cv2.putText(frame, 
                "FPS: {}".format(framerate), 
                (10, 50), 
                font, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA)
    
    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
