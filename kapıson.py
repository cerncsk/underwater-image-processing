# xd

import cv2
import numpy as np
import math


def nothing(a):
    pass


cv2.namedWindow('controls', 2)
# create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('r', 'controls', 0, 255, nothing)
cv2.createTrackbar('g', 'controls', 0, 255, nothing)
cv2.createTrackbar('b', 'controls', 0, 255, nothing)
cv2.createTrackbar('h', 'controls', 0, 255, nothing)
cv2.createTrackbar('s', 'controls', 0, 255, nothing)
cv2.createTrackbar('v', 'controls', 0, 255, nothing)

#cap = cv2.VideoCapture("C:\\Users\\cloudX\\Desktop\\kapi2.avi")


#cap = cv2.VideoCapture("http://192.168.2.2:8080/?action=stream")
cap = cv2.VideoCapture(0)

#
# cap.set(3, 1290)
# cap.set(4, 948)


def getContours(img, imgContour, areaList, cords, data, areas):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
        if area < 300000:
            peri = cv2.arcLength(cnt, True)  # konturun çevresini hesaplar.
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, closed=False)

            cv2.drawContours(img, [approx], -1, (255, 255, 255), 2)

            if area > 1000 and 4 <= len(approx) <= 6:
                n = approx.ravel()
                i = 0

                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]

                        # String containing the co-ordinates.
                        string = str(x) + " " + str(y)

                        if (i == 0):
                            # text on topmost co-ordinate.
                            cv2.putText(imgContour, string, (x, y),
                                        font, 1, (255, 0, 0), 4)
                        else:
                            # text on remaining co-ordinates.
                            cv2.putText(imgContour, string, (x, y),
                                        font, 1, (0, 255, 0), 4)
                    i = i + 1

                    m = ((n[3] - n[1]) ** 2 + (n[2] - n[0]) ** 2) ** 0.5
                    m1 = ((n[5] - n[7]) ** 2 + (n[4] - n[6]) ** 2) ** 0.5

                    m2 = ((n[1] - n[7]) ** 2 + (n[0] - n[6]) ** 2) ** 0.5
                    m3 = ((n[3] - n[5]) ** 2 + (n[2] - n[4]) ** 2) ** 0.5

                    m4 = ((n[2] - n[4]) ** 2 + (n[3] - n[5]) ** 2) ** 0.5
                    m5 = ((n[0] - n[2]) ** 2 + (n[1] - n[3]) ** 2) ** 0.5

                    if 0.9719989395702574 < (m / m1) < 1.8135595674011302 or 0.9297857054608672 < (
                            m2 / m3) < 1.9997857054608672:

                        if 1.9019989395702574 < (m / m4) < 2.9019989395702574 or 1.9019989395702574 < (
                                m2 / m5) < 2.9019989395702574:
                            M = cv2.moments((cnt))
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            print("x koordinatı:")
                            print(cX)
                            print("y koordinatı")
                            print(cY)

                            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2)

                            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                            cv2.putText(frame, "center", (cX - 20, cY - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_PLAIN
    if ret:
        # img = cv2.flip(frame, 1)
        b, g, r = cv2.split(frame)
        # cv2.imshow('r channel',r)
        # cv2.imshow('g channel',g)
        # cv2.imshow('b channel',b)

        # r2=float(cv2.getTrackbarPos('r','controls'))
        # g2 = float(cv2.getTrackbarPos('g', 'controls'))
        # b2 = float(cv2.getTrackbarPos('b', 'controls'))
        r = r + 103
        g = g + 44
        b = b + 255

        merged = cv2.merge([r, g, b])
        cv2.imshow("merge", merged)
        # cv2.imshow("frame", frame)

        hsv = cv2.cvtColor(merged, cv2.COLOR_BGR2HSV)  # convert it to hsv

        h, s, v = cv2.split(hsv)
        h = h + 141
        s = s + 47
        v = v + 255
        final_hsv = cv2.merge((h, s, v))

        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("img", img)
        # bgr = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 9)
        th3 = cv2.adaptiveThreshold(blurred, 260, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow("th3", th3)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(th3, kernel, iterations=3)
        dilation = cv2.dilate(erosion, kernel, iterations=3)
        cv2.imshow("dilation", dilation)

        # gray = np.float32(dilation)
        # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # # result is dilated for marking the corners, not important
        # # dst = cv2.dilate(dst, None)
        # # Threshold for an optimal value, it may vary depending on the image.
        # img[dst > 0.01 * dst.max()] = [0, 0, 255]
        # cv2.imshow('dst', img)
        # print(dst)
        imgContour = img.copy()
        hull_list = []
        areaList = []
        cords = []
        center = []
        data = []
        areas = []
        getContours(th3, imgContour, areaList, cords, data, areas)
        cv2.imshow("Contour", imgContour)
        cv2.imshow("frame", frame)

    else:

        #cap = cv2.VideoCapture("C:\\Users\\cloudX\\Desktop\\kapi2.avi")

        cap = cv2.VideoCapture("http://192.168.2.2:8080/?action=stream")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break