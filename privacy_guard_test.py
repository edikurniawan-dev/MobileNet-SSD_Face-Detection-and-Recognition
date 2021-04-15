from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os

def detect_and_predict_mask(frame, faceNet, face_model):
	(h, w) = frame.shape[:2]
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
	blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []
	locs = []
	preds = []

	pred = {"[0]": "edi", "[1]": "habib", "[2]": "unknown"}
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			# box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			# (startX, startY, endX, endY) = box.astype("int")
			#
			# # ensure the bounding boxes fall within the dimensions of
			# # the frame
			# (startX, startY) = (max(0, startX), max(0, startY))
			# (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 1)
			face = frame[startY:endY, startX:endX]
			# face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
			# face = img_to_array(face)
			# face = preprocess_input(face)

			face = face / 255
			face = face.reshape(1, 224, 224, 3)

			# face = face.reshape(1, 150528)
			# print(face)
			# 	input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
			# 	input_im = input_im / 255
			# 	input_im = input_im.reshape(1, 224, 224, 3)

			# faces.append(face)
			# locs.append((startX, startY, endX, endY))

			# face_crop = frame[startY:endY, startX:endX]
			# cv2.imwrite('dataset/temp/face' + str(count) + '.jpg', face_crop)
			# print(face)
# only make a predictions if at least one face was detected
	# if len(faces) > 0:
			preds = np.argmax(face_model.predict(face, 1, verbose=0), axis=1)
			facial = pred[str(preds)]
			print(facial)
		# preds = face_model.predict(faces, batch_size=32)

	# return preds


# load our serialized face detector model from disk
prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
weights_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

# load the face mask detector model from disk
model_recog = load_model("Facial_recog.h5")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
pred = {"[0]": "edi", "[1]": "habib", "[2]": "unknown"}
count = 0
while True:
	frame = vs.read()
	flip_frame = cv2.flip(frame, 1)
	frame = imutils.resize(flip_frame, width=720)

	detect_and_predict_mask(frame, faceNet, model_recog)
	# count = count + 1
	# loop over the detected face locations and their corresponding
	# locations
	# facial = pred[str(prediction)]
	# print(facial)

	# for (box, pred) in zip(locs, preds):
	# 	# unpack the bounding box and predictions
	# 	(startX, startY, endX, endY) = box
	# 	# (edi, habib, unknown) = pred
	# 	print(box)
	# 	print(preds)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

# import os
# import cv2
# import numpy as np
#
#
# facial_recog_dict = {"[0]": "edi",
# 					 "[1]": "habib",
# 					 "[2]": "unknown"}
#
#
# def draw_test(pred, i, path_image):
# 	facial = facial_recog_dict[str(pred)]
# 	print(path_image)
# 	print(str(i) + ". " + facial)
# 	# print("\n")
#
#
# path = "dataset/test_crop/"
# many_file_in_dict = len(os.listdir(path))
# image_file = os.listdir(path)
# # print(image_file)
#
# # for i in range(many_file_in_dict):
# path_image = os.path.join(path + image_file[0])
# input_im = cv2.imread(path_image)
#
# # print(input_im)
# # input_original = input_im.copy()
# # input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
#
# input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
# input_im = input_im / 255
# input_im = input_im.reshape(1, 224, 224, 3)
#
# print(input_im)
	# res = np.argmax(model_recog.predict(input_im, 1, verbose=0), axis=1)

	# draw_test(res, i + 1, path_image)
# # cv2.waitKey(0)

cv2.destroyAllWindows()