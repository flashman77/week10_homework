import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("/home/flashman77/week10/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/home/flashman77/week10/haarcascade_eye.xml")

cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # first camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # camera width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # camera height

while (True):
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    print("Number of faces detected: " + str(len(faces)))

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # end by ESC
        break

cap.release()
cv2.destroyAllWindows()