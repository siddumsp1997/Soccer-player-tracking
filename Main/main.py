

# Standard packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# Custom packages
from lib.video_source import getVideoSource
from lib.fgbg_calculations import getThresholdedFrame
from lib.heatmap import Heatmap
from lib.polygon import drawQuadrilateral
from lib.coordinate_transform import windowToFieldCoordinates

# init frame count 
frameCount = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", help="path to the video file")

args = vars(ap.parse_args())


# get the video from the parsed argument
video = getVideoSource(args)


# get BackgroundSubtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

padding = 20
# get top-view field dimensions
width = 280
height = 334


# minimum area of detected object(s)
# To differentiate football and the player
objectMinArea = width*0.1 * height*0.2


# create a black image/frame where the top-view field will be drawn
field = np.zeros((height + padding*2,width + padding*2,3), np.uint8)

# top-view rectangle coordinates
(xb1, yb1) = (padding, padding)
(xb2, yb2) = (padding + width, padding)
(xb3, yb3) = (padding + width, padding + height)
(xb4, yb4) = (padding, padding + height)


# crea heatmap object
heatmap = Heatmap(field, width, height)

# Set of coordinates of the end pts of the frame
coords = []

# loop over the frames of the video
while True:
	# grab the current frame
	(ReadorNot, frame) = video.read()

	# if the frame could not be ReadorNot, then end is reached
	if not ReadorNot:
		break

	# To reduce CPU load, resize the frame
	frame = imutils.resize(frame, width=800)

	# increase frame count
	frameCount = frameCount + 1

	# Initialise the coords with the end points of the frame in the beginning itself
	if frameCount == 1:
		
		coords.append((0,0))
		coords.append((799,0))
		coords.append((799,449))
		coords.append((0,449))
	

	# draw perspective field
	drawQuadrilateral(frame, coords, 0, 255, 0, 2)

	# apply color subtractions and calculations to get a black and white frame
	# making it possible for computer to recognize clear contours
	thresh = getThresholdedFrame(fgbg, frame)

	# get the contours of all white regions in the frame
	(image, contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop over all contours
	for contour in contours:
		# if the contour is too small, ignore it
		if cv2.contourArea(contour) < objectMinArea:
			continue

		# compute the bounding box for the contour, draw it on the frame
		(x, y, w, h) = cv2.boundingRect(contour)
		basePoint = ((x + (w/2)), (y + h))

		# get the top-view relative coordinates
		(xbRel, ybRel) = heatmap.getPosRelativeCoordinates(basePoint, coords)

		# draw rectangle around the detected object and a red point in the center of its base
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		# get the top-view absolute coordinates
		(xb, yb) = heatmap.getPosAbsoluteCoordinates((xbRel, ybRel), (xb1, yb1))

		# draw overlayed opacity circle every 5 frames
		if frameCount % 5 == 0:
			heatmap.drawOpacityCircle((xb, yb), 255, 0, 0, 0, 15)

	# display the windows
	cv2.imshow('Rendered output',frame)
	cv2.imshow('Heat map',thresh)
	cv2.imshow('Foot print', field)


	def leftClickDebug(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			resultCoord = windowToFieldCoordinates((x, y), coords, width, height)
			print "Coordinates to real coordinates", resultCoord

	# wait for key press
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# release the video source and close opened windows
video.release()
cv2.destroyAllWindows()