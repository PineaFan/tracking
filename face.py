import cv2
import time
import math
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaces = mp.solutions.face_mesh
face = mpFaces.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mpDraw = mp.solutions.drawing_utils

def render(results):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, faceLms, mpFaces.FACEMESH_CONTOURS)
    cv2.imshow("Image", img)

def fromList(l, a):
    return [l[i] for i in a]

def getCoords():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    landmarks = results.multi_face_landmarks
    if landmarks:
        return landmarks[0].landmark, results
    return landmarks, results

precision = 1000
while True:
    landmarks, results = getCoords()
    if landmarks:
        points = [[p.x, p.y, p.z, 2] for p in landmarks]
        origin = points[0]  # Make this the center
        for index, point in enumerate(points):  # For each point
            points[index] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]  # Translate such that point 0 is at (0, 0, 0)
            points[index] = [round(points[index][0] * precision), round(points[index][1] * precision), round(points[index][2] * precision)]  # Scale to a reasonable size
    render(results)

    cv2.waitKey(1)

