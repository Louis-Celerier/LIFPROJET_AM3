import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("ressources/shape_predictor_68_face_landmarks.dat")
#src = cv2.imread('ressources/training-originals/0002_1.jpg', 0)
src = cv2.imread('ressources/training-originals/0000_00000001.jpg')
target = cv2.imread('ressources/training-originals/0001_00000001.jpg', 0)

width = 640
height = 360
src = cv2.resize(src, (width, height))
gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
target = cv2.resize(target, (width, height))

mask = np.zeros_like(gray_src)

face = detector(gray_src)
for point in face:
    landmarks = predictor(gray_src, point)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
        #cv2.circle(src, (x, y), 4, (255, 0, 0), -1)

    points = np.array(points, np.int32)
    convexHull = cv2.convexHull(points)
    #cv2.polylines(src, [convexHull], True, (255,0,0), 3)
    # Creating mask
    cv2.fillConvexPoly(mask, convexHull, 255)

    face_src = cv2.bitwise_and(src, src, mask=mask)

    #Triangulation
    rect = cv2.boundingRect(convexHull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for t in triangles:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])
        
        cv2.line(face_src, p1, p2, (255,0,0), 2)
        cv2.line(face_src, p3, p2, (255,0,0), 2)
        cv2.line(face_src, p1, p3, (255,0,0), 2)

#Showing img
while True:
    cv2.imshow("mask", face_src)
    key = cv2.waitKey(1)
    if key == 27:
        break