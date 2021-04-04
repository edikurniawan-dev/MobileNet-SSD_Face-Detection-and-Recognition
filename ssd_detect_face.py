from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pynput.keyboard import Key, Controller
import win32gui
import win32con

def detect_face(frame, faceNet):
	# mendefinisikan ukuran frame dan buat kedalam tipe blob

	(h, w) = frame.shape[:2]
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)


	# memasukkan tipe frame dalam bentuk blob ke neural net untuk mendeteksi wajah
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# inisialisasi list wajah, lalu menentukan lokasi (kotak")
	# inisialisasi list prediksi dari neural net face sebelumnya
	faces = []
	locs = []
	preds = []

	# looping untuk deteksi
	for i in range(0, detections.shape[2]):
		# ekstrak tingkat kepercayaan (probabilitas dan akurasi)
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	return(locs, preds, faces)

def predict_face():
	# for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
	(startX, startY, endX, endY) = box
	(edi, unknown) = pred

	# determine the class label and color we'll use to draw
	# the bounding box and text


	return(label_name)

# load our serialized face detector model from disk
# prototxt_path = r"face-detector-model/other-model-face-detector.prototxt"
# weights_path = r"face-detector-model/res10_300x300_ssd_iter_140000.caffemodel"

# prototxt_path = 'MobileNet-SSD-master/other-model-face-detector.prototxt'
# weights_path = 'MobileNet-SSD-master/mobilenet_iter_73000.caffemodel'

prototxt_path = 'MobilenetSSDFace-master/models/deploy/ssd-face.prototxt'
weights_path = 'MobilenetSSDFace-master/models/deploy/ssd-face.caffemodel'

face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# load the face mask detector model from disk
# maskNet = load_model("face_recognition4.model")
face_recog = load_model("facerecog-anime.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

count = 0
counter = 0
# loop over the frames from the video stream

prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	flip_frame = cv2.flip(frame, 1)
	frame = imutils.resize(flip_frame, width=540)

	(locs, preds, faces) = detect_face(frame, face_net)

	(startX, startY, endX, endY) = box
	(edi, unknown) = pred

	label_name = "Face"
	# color = (0, 255, 0) if label_name == "Edi" else (0, 0, 255)

	label = "{}: {:.2f}%".format(label_name, max(label_name) * 100)
	label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

	cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
	cv2.putText(frame, label, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	# faces = np.array(faces, dtype="float32")
	# # preds = face_recog.predict(faces, batch_size=32)
	# preds = face_recog.predict(faces, batch_size=32)

	for (box, pred) in zip(locs, preds):
		predict_face()

	# show the output frame
	new_frame_time = time.time()

	# Calculating the fps

	# fps will be number of frame processed in given time frame
	# since their will be most of time error of 0.001 second
	# we will be subtracting it to get more accurate result
	fps = 1 / (new_frame_time - prev_frame_time)
	prev_frame_time = new_frame_time

	# converting the fps into integer
	fps = int(fps)

	# converting the fps to string so that we can display it on frame
	# by using putText function
	fps = str(fps)
	cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q") or key == ord("Q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()