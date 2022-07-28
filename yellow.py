#detect yellow objects(yellow stick)

import numpy as np
import cv2

cap = cv2.VideoCapture("C:\\Users\\cloudX\\Desktop\\video9.mp4")
#cap = cv2.VideoCapture("http://192.168.2.2:8080/?action=stream")


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

while (True):
    # Capture frame-by-frame
    read = cap.read()
    ret, frame = read
    font = cv2.FONT_HERSHEY_PLAIN
    if ret:
        b, g, r = cv2.split(frame)
        r = r * cv2.getTrackbarPos('r', 'controls')
        g = g * cv2.getTrackbarPos('g', 'controls')
        b = b * cv2.getTrackbarPos('b', 'controls')

        merged = cv2.merge([r, g, b])
        cv2.imshow("merge", merged)
        # cv2.imshow("frame", frame)

        hsv = cv2.cvtColor(merged, cv2.COLOR_BGR2HSV)  # convert it to hsv

        h, s, v = cv2.split(hsv)
        h = h + cv2.getTrackbarPos('h', 'controls')
        s = s + cv2.getTrackbarPos('s', 'controls')
        v = v + cv2.getTrackbarPos('v', 'controls')
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("img", img)

        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.medianBlur(gray, 5)

        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3,5)
        cv2.imshow("th3",th3)
        kernel = np.ones((5, 5), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)

        gray = cv2.dilate(gray, kernel, iterations=1)
        cv2.imshow("gray",gray)

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
        """
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 20, param1=50, param2=30, minRadius=0, maxRadius=80)

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle in the image
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                print("x=",x)
                print("y=",y)

            # Display the resulting frame
            cv2.imshow('gray', gray)
        cv2.imshow('frame', output)
        """
    else:

        #cap = cv2.VideoCapture("http://192.168.2.2:8080/?action=stream")
        cap = cv2.VideoCapture("C:\\Users\\cloudX\\Desktop\\video9.mp4")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
