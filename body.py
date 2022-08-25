import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
solution = mp.solutions.holistic
detect = solution.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils


def render(results):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detect.process(imgRGB)
    mpDraw.draw_landmarks(img, results.pose_landmarks, solution.POSE_CONNECTIONS)

    cv2.imshow("Image", img)


def getCoords():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detect.process(imgRGB)
    landmarks = results.pose_landmarks
    return landmarks, results


while True:
    landmarks, results = getCoords()
    render(results)

    cv2.waitKey(1)

