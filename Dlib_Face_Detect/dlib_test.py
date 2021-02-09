import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("ressources/shape_predictor_68_face_landmarks.dat")
#src = cv2.imread('ressources/training-originals/0002_1.jpg', 0)
src = cv2.imread('ressources/training-originals/0000_00000001.jpg', 0)

width = 640
height = 360
img = cv2.resize(src, (width, height))

face = detector(img)
for point in face:
    x1 = point.left()
    x2 = point.right()
    y1 = point.top()
    y2 = point.bottom()
    #cv2.rectangle(img, (x1,y1), (x2, y2), (255,255,255), 4)

    landmarks = predictor(img, point)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
#Showing img
while True:
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        break