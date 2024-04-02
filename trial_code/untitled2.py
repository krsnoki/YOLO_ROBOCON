import cv2 as cv
import numpy as np



VideoCapture = cv.VideoCapture(0)

while True:
    ret, frame = VideoCapture.read()
    if not ret: 
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

    cv.imshow('blur frame', blurFrame)

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

VideoCapture.release()
cv.destroyAllWindows()