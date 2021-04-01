from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="dataset/edi", help="path to output directory")
args = vars(ap.parse_args())
path = args["output"]
path_arr = path.split('/')

# go into desired directory for saving images
# os.chdir(path_arr[0] + '/' + path_arr[1])
os.chdir(path)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(usePiCamera=True).start() # if using RPi3 camera!
vs = VideoStream(src=0).start()
# time.sleep(2.0)


def cropFace():
	image = cv2.imread('image.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=3,
		minSize=(30, 30)
	)

	print("[INFO] Found {0} Faces.".format(len(faces)))

	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = image[y:y + h, x:x + w]
		print("[INFO] Object found. Saving locally.")
		cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

	os.remove('image.jpg')

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	flip_frame = cv2.flip(frame, 1)
	# frame = imutils.resize(frame, width=400)

	# show the output frame
	cv2.imshow("Frame", flip_frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `spacebar` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == 32:
		cv2.imwrite('image.jpg', flip_frame)
		print("Image captured!")
		cropFace()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()