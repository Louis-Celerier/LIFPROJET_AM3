import cv2
import numpy as np
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("ressources/shape_predictor_81_face_landmarks.dat")
src = cv2.imread('ressources/training-originals/0000_00000001.jpg')
cam = cv2.VideoCapture(0)


def get_landmarks(face, gray_img):
    landmarks = predictor(gray_img, face)
    points_landmarks = []
    for n in range(0, landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points_landmarks.append((x, y))
    return points_landmarks

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def triangulation_point(triangle_id, points_landmarks, points, target=False):
    t_p1 = points_landmarks[triangle_id[0]]
    t_p2 = points_landmarks[triangle_id[1]]
    t_p3 = points_landmarks[triangle_id[2]]
    triangle = np.array([t_p1, t_p2, t_p3], np.int32)

    (x, y, w, h) = cv2.boundingRect(triangle)        
    cropped_t_mask = np.zeros((h, w), np.uint8)

    points = np.array([[t_p1[0] - x, t_p1[1] - y],
                       [t_p2[0] - x, t_p2[1] - y],
                       [t_p3[0] - x, t_p3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_t_mask, points, 255)

    if not target:
        cropped_triangle = src[y: y + h, x: x + w]
        return (points, cropped_triangle)
    else:
        return (points, cropped_t_mask, x, y, w, h)
        

gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(gray_src)

# Source
faces = detector(gray_src)
for f in faces:
    points_landmarks = get_landmarks(f, gray_src)
    points = np.array(points_landmarks, np.int32)
    convexhull = cv2.convexHull(points)
    
    # Creation du mask
    cv2.fillConvexPoly(mask, convexhull, 255)

    face_src = cv2.bitwise_and(src, src, mask=mask)

    # Triangulation de Delaunay
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points_landmarks)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    triangles_ids = []
    for t in triangles:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])

        id_p1 = np.where((points == p1).all(axis=1))
        id_p1 = extract_index_nparray(id_p1)

        id_p2 = np.where((points == p2).all(axis=1))
        id_p2 = extract_index_nparray(id_p2)

        id_p3 = np.where((points == p3).all(axis=1))
        id_p3 = extract_index_nparray(id_p3)
        
        if id_p1 is not None and id_p2 is not None and id_p3 is not None:
            triangles_ids.append([id_p1, id_p2, id_p3])

# Target
points_landmarks2 = None
while True:
    _,target = cam.read()
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    height, width, channels = target.shape
    new_face = np.zeros((height, width, channels), np.uint8)
    faces2 = detector(gray_target)
    for f in faces2:
        points_landmarks2 = get_landmarks(f, gray_target)
        points2 = np.array(points_landmarks2, np.int32)
        convexhull2 = cv2.convexHull(points2)
    if points_landmarks2 == None:
        continue
    # Triangulation des 2 faces
    for triangle_id in triangles_ids:
        # 1ere Face
        (points, cropped_triangle) = triangulation_point(triangle_id, points_landmarks, points)

        # 2eme Face
        (points2, cropped_t2_mask, x, y, w, h) = triangulation_point(triangle_id, points_landmarks2, points2, True)

        # Deformation des triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        matrix = cv2.getAffineTransform(points, points2)
        triangle_warped = cv2.warpAffine(cropped_triangle, matrix, (w, h))
        triangle_warped = cv2.bitwise_and(triangle_warped, triangle_warped, mask=cropped_t2_mask)

        # Reconstruction des points
        new_face_rect_area = new_face[y: y + h, x: x + w]
        gray_new_face_rect_area = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
        gray_new_face_rect_area, mask_triangles_designed = cv2.threshold(gray_new_face_rect_area, 1, 255, cv2.THRESH_BINARY_INV)
        triangle_warped = cv2.bitwise_and(triangle_warped, triangle_warped, mask=mask_triangles_designed)
        new_face_rect_area = cv2.add(new_face_rect_area, triangle_warped)
        new_face[y: y + h, x: x + w] = new_face_rect_area

    # FaceSwaping
    mask_target_face = np.zeros_like(gray_target)
    mask_target_head = cv2.fillConvexPoly(mask_target_face, convexhull2, 255)
    mask_target_face = cv2.bitwise_not(mask_target_head)

    noface_target_head = cv2.bitwise_and(target, target, mask=mask_target_face)
    result = cv2.add(noface_target_head, new_face)
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    target_with_color = cv2.seamlessClone(result, target, mask_target_head, center_face, cv2.MIXED_CLONE)


    cv2.imshow("FaceSwap", target_with_color)
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
