import math
import time

import alsaaudio
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cap = cv2.VideoCapture(0)
m = alsaaudio.Mixer()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mpDraw = mp.solutions.drawing_utils

def render(results):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Image", img)

def matPlotRender(fig, results, normal):
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    x_values = [r[0] for r in results]
    y_values = [r[1] for r in results]
    z_values = [r[2] for r in results]

    ax.scatter(0, 0, 0, c='#000000', marker='o')
    ax.text3D(0, 0, 0, 'Palm')
    ax.scatter(x_values[1:5], y_values[1:5], z_values[1:5], c='#FF0000', marker='o')
    ax.text3D(x_values[4], y_values[4], z_values[4], 'Thumb')
    ax.scatter(x_values[5:9], y_values[5:9], z_values[5:9], c='#00FF00', marker='o')
    ax.text3D(x_values[8], y_values[8], z_values[8], 'Index')
    ax.scatter(x_values[9:13], y_values[9:13], z_values[9:13], c='#0000FF', marker='o')
    ax.text3D(x_values[12], y_values[12], z_values[12], 'Middle')
    ax.scatter(x_values[13:17], y_values[13:17], z_values[13:17], c='#FFFF00', marker='o')
    ax.text3D(x_values[16], y_values[16], z_values[16], 'Ring')
    ax.scatter(x_values[17:], y_values[17:], z_values[17:], c='#FF00FF', marker='o')
    ax.text3D(x_values[20], y_values[20], z_values[20], 'Pinky')

    ax.scatter(200, 200, 200, c='#000000', marker='o')
    ax.scatter(-200, -200, -200, c='#000000', marker='o')

    centroid = [
        (x_values[5] + x_values[17]) / 3,
        (y_values[5] + y_values[17]) / 3,
        (z_values[5] + z_values[17]) / 3
    ]

    xv = [0 + centroid[0], (normal[0] / 100) + centroid[0]]
    yv = [0 + centroid[1], (normal[1] / 100) + centroid[1]]
    zv = [0 + centroid[2], (normal[2] / 100) + centroid[2]]

    ax.plot(xv, yv, zv, c='#FF0000')

    for i in range(0, 5): # For each finger
        for j in range(0, 3): # For each joint
            cols = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
            xv = [x_values[(i * 4) + j + 1], x_values[(i * 4) + j + 2]]
            yv = [y_values[(i * 4) + j + 1], y_values[(i * 4) + j + 2]]
            zv = [z_values[(i * 4) + j + 1], z_values[(i * 4) + j + 2]]
            ax.plot(xv, yv, zv, c=cols[i])

    vert = [list(zip(np.array([x_values[0], x_values[5], x_values[17]]), np.array([y_values[0], y_values[5], y_values[17]]), np.array([z_values[0], z_values[5], z_values[17]])))]
    srf = Poly3DCollection(vert, alpha=.25, facecolor='#800000')
    ax.add_collection3d(srf)

    ax.zaxis.set_tick_params(labelsize=10)
    ax.set_xlabel('X')
    plt.xlim([200, -200])
    ax.xaxis.set_tick_params(labelsize=10)
    ax.set_ylabel('Y')
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_zlabel('Z')
    plt.show()


def fromList(l, a):
    return [l[i] for i in a]

def getFingerCoords():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    landmarks = results.multi_hand_landmarks
    if landmarks:
        return landmarks[0].landmark, results
    return landmarks, results

precision = 1000
fig = plt.figure()
plt.ion()
maxDist = .001
minDist = 1000000
while True:
    landmarks, results = getFingerCoords()
    if landmarks:
        points = [[p.x, p.y, p.z, 2] for p in landmarks]
        origin = points[0]  # Make this the center
        for index, point in enumerate(points):  # For each point
            points[index] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]  # Translate such that point 0 is at (0, 0, 0)
            points[index] = [round(points[index][0] * precision), round(points[index][1] * precision), round(points[index][2] * precision)]  # Scale to a reasonable size

        # Find the distance between points 4 and 8
        dist = math.sqrt(math.pow(points[4][0] - points[8][0], 2) + math.pow(points[4][1] - points[8][1], 2) + math.pow(points[4][2] - points[8][2], 2))
        if dist > maxDist:
            maxDist = dist
        if dist < minDist:
            minDist = dist

        try:
            # Scale dist based on min and max to get a value between 0 and 100
            dist = (dist - minDist) / (maxDist - minDist) * 100 - 10
            dist = round(min(max(dist * 1.2, 0), 100))
            m.setvolume(dist)
            print(dist)
        except:
            pass

        # Turn 5 and 17 into vectors B and C
        B = np.array(points[5])
        C = np.array(points[17])
        # # Find the normal of the plane
        normal = np.cross(B, C)

        matPlotRender(fig, points, list(normal))
        # break
    render(results)

    cv2.waitKey(1)

