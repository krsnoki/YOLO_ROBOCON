import cv2
import numpy as np

img = cv2.imread('frc1195.jpg')
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)

cv2.destroyAllWindows()
# The code above reads an image, converts it to grayscale, and applies a median blur to reduce noise. It then uses the Hough Circle Transform to detect circles in the image. The detected circles are drawn on the image using OpenCV's circle drawing functions. Finally, the image with the detected circles is displayed.