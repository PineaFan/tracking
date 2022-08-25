import math
import time

import alsaaudio
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cap = cv2.VideoCapture(0)

mpFaces = mp.solutions.face_mesh
face = mpFaces.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)
mpDraw = mp.solutions.drawing_utils

def matPlotRender(fig, results, lside, rside):
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    x_values = [r[0] for r in results]
    y_values = [r[1] for r in results]
    z_values = [r[2] for r in results]

    ax.scatter(0, 0, 0, c='#000000', marker='o')
    ax.plot([lside[0], rside[0]], [lside[1], rside[1]], [0, 0], c='#0000FF')

    ax.zaxis.set_tick_params(labelsize=10)
    ax.set_xlabel('X')
    plt.xlim([200, -200])
    ax.xaxis.set_tick_params(labelsize=10)
    ax.set_ylabel('Y')
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_zlabel('Z')
    plt.show()


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


def getCoords():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    landmarks = results.multi_face_landmarks
    if landmarks:
        return landmarks[0].landmark, results
    return landmarks, results


def clamp(n):
    return max(0, min(100, n))


straight = None
lastChange = None
precision = 10000
strength = 10
m = alsaaudio.Mixer()
v = m.getvolume()
m.setvolume(round((v[0] + v[1]) / 2), channel=0)
m.setvolume(round((v[0] + v[1]) / 2), channel=1)
while True:
    m = alsaaudio.Mixer()
    v = m.getvolume()
    if lastChange:
        v[0] -= lastChange[0]
        v[1] -= lastChange[1]
    landmarks, results = getCoords()
    if landmarks:
        points = [[p.x, p.y, p.z] for p in landmarks]
        origin = points[0]  # Make this the center
        for index, point in enumerate(points):  # For each point
            points[index] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]  # Translate such that point 0 is at (0, 0, 0)
            points[index] = [round(points[index][0] * precision), round(points[index][1] * precision), round(points[index][2] * precision)]  # Scale to a reasonable size
            points[index][1], points[index][2] = points[index][2], -points[index][1]
        lside = [points[234][0], points[234][1]]
        rside = [points[454][0], points[454][1]]
        gradient = (rside[1] - lside[1]) / ((rside[0] - lside[0]) or 0.001)
        if straight is None:
            straight = gradient
        gradient = gradient - straight
        diff = [clamp(round(v[0] - math.sin(math.atan(gradient)) * strength)), clamp(round(v[1] + math.sin(math.atan(gradient)) * strength))]
        print(diff)
        m.setvolume(diff[0], channel=0)
        m.setvolume(diff[1], channel=1)
        lastChange = [diff[0] - v[0], diff[1] - v[1]]
    # render(results)
    cv2.waitKey(1)
