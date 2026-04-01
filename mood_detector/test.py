import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=False)
    print(result)
else:
    print('Camera failed')