import numpy as np
import cv2
import time

# initialize the video capture object
w = 1000
h = 1000
my_camera = cv2.VideoCapture(2)
my_camera.set(3, w)
my_camera.set(4, h)
time.sleep(2)

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

while True:
    ret, frame = my_camera.read()
    if not ret:
        print("There is no more frame to read, exiting...")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # convert from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_frame', hsv_frame)

    # lower and upper limits for the blue color
    lower_limit = np.array([99, 135, 51])
    upper_limit = np.array([116, 226, 255])

    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    bbox = cv2.boundingRect(mask)

    # if we get a bounding box, use it to draw a rectangle on the image
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("Object not detected")

    cv2.imshow('frame', frame)
    
    # write the frame to the output file
    output.write(frame)
    
    if cv2.waitKey(30) == ord('q'):
        break

# Release video capture and writer
my_camera.release()
output.release()
cv2.destroyAllWindows()
