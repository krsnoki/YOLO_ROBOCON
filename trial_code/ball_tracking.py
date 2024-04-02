''' USAGE
python trackballs.py --video example.mp4
python trackballs.py
'''

import numpy as np
import argparse
import cv2

def traceCircle(frame, mask):
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)


        # only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle on the frame
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)

		
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])


#defining lower and upper boundaries in HSV
greenLower = (44, 86, 56)
greenUpper = (94, 246, 255)

redLower = (000, 102, 84)
redUpper = (255, 248, 135)

while True:
	(grabbed, frame) = camera.read()

	# to check if video has ended
	if args.get("video") and not grabbed:
		break

	# resize the frame, and convert it to the HSV color space
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to blobs from mask
	
	mask1 = cv2.inRange(hsv, greenLower, greenUpper)
	mask1 = cv2.erode(mask1, None, iterations=2)
	mask1 = cv2.dilate(mask1, None, iterations=2)

	traceCircle(frame,mask1)
	
	#constructing mask for the colro red 
	
	mask2 = cv2.inRange(hsv, redLower, redUpper)
	mask2 = cv2.erode(mask2, None, iterations=2)
	mask2 = cv2.dilate(mask2, None, iterations=2)
	
	traceCircle(frame,mask2)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()