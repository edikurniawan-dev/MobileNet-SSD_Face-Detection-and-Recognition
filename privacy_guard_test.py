# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os

def detect_and_predict_face(frame, faceNet, face_model):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
	# blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	pred = {"[0]": "edi", "[1]": "habib", "[2]": "unknown"}
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 1)
			face_detect = frame[startY:endY, startX:endX]
			# face_detect = cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB)
			face_detect = cv2.resize(face_detect, (224, 224), interpolation=cv2.INTER_LINEAR)
			# face = img_to_array(face)
			# face = preprocess_input(face)

			face_detect = face_detect / 255
			face_detect = face_detect.reshape(1, 224, 224, 3)


			#
			# faces.append(frame)
			# locs.append((startX, startY, endX, endY))
# only make a predictions if at least one face was detected
# 	if len(faces) > 0:
# 		# faces = np.array(faces, dtype="float32")
# 		for i in range(len(faces)):
			preds = np.argmax(face_model.predict(face_detect, 1, verbose=0), axis=1)
			facial = pred[str(preds)]
			# print(facial)
		# preds = face_model.predict(faces, batch_size=32)
			cv2.putText(frame, facial, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


	# return (locs, faces, facial)


# load our serialized face detector model from disk
prototxt_path = "face-detector-model/ssd-model/deploy.prototxt"
weights_path = "face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel"

# prototxt_path = "face-detector-model/mobnet-ssd-model/ssd-face.prototxt"
# weights_path = "face-detector-model/mobnet-ssd-model/ssd-face.caffemodel"
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

	detect_and_predict_face(frame, faceNet, model_recog)
	# count = count + 1
	# loop over the detected face locations and their corresponding
	# locations
	# facial = pred[str(prediction)]
	# print(facial)

	# for (box, face) in zip(locs, faces):
	# 	# unpack the bounding box and predictions
	# 	(startX, startY, endX, endY) = box
	# 	# (edi, habib, unknown) = pred
	# 	# cv2.rectangle(frame, (startX, startY), (endX, endY), 0, 255, 0, 2)
	# 	# cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
	# 	cv2.putText(frame, facial, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


		# preds = np.argmax(face_model.predict(face_detect, 1, verbose=0), axis=1)

	# print(box)
		# print(preds)

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
# for i in range(many_file_in_dict):
# 	path_image = os.path.join(path + image_file[i])
# 	input_im = cv2.imread(path_image)
#
# 	# print(input_im)
# 	# input_original = input_im.copy()
# 	# input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
#
# 	input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
# 	input_im = input_im / 255
# 	input_im = input_im.reshape(1, 224, 224, 3)
#
# 	# print(input_im)
# 	res = np.argmax(model_recog.predict(input_im, 1, verbose=0), axis=1)
#
# 	draw_test(res, i + 1, path_image)
# # cv2.waitKey(0)

cv2.destroyAllWindows()