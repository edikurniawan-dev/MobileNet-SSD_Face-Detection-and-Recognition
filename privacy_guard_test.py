from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import time
from pynput.keyboard import Key, Controller
import win32gui
import win32con

def detect_and_predict_face(frame, faceNet, face_model):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	# blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []

	pred = {"[0]": "edi","[1]": "unknown"}
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face_detect = frame[startY:endY, startX:endX]
			face_detect = cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB)
			face_detect = cv2.resize(face_detect, (224, 224), interpolation=cv2.INTER_LINEAR)
			# face = img_to_array(face)
			# face = preprocess_input(face)

			face_detect = face_detect / 255
			face_detect = face_detect.reshape(1, 224, 224, 3)
			# face_detect = preprocess_input(face_detect)
			faces.append(face_detect)

			preds = np.argmax(face_model.predict(face_detect, 1, verbose=0), axis=1)
			facial = pred[str(preds)]

			if facial == "edi":
				label = "Edi"
				color = (0, 255, 0)
			# if facial == "habib":
			# 	label = "Habib"
			# 	color = (0, 255, 0)
			if facial == "unknown":
				label = "Unknown"
				color = (0, 0, 255)

			# print(facial)

			# label = "{}: {:.2f}%".format(label)
			label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
			cv2.putText(frame, label, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))


# print("[INFO] Found {0} Faces. ".format(len(faces)), count)
# faces = np.array(faces, dtype="float32")
# preds = face_recog.predict(faces, batch_size=32)
# label_name = predict_face()
	if len(faces) == 0:
		return(faces == [], label == "none")
# load our serialized face detector model from disk
prototxt_path = "face-detector-model/ssd-model/deploy.prototxt"
weights_path = "face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel"

# prototxt_path = "face-detector-model/mobnet-ssd-model/ssd-face.prototxt"
# weights_path = "face-detector-model/mobnet-ssd-model/ssd-face.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

# load the face mask detector model from disk
model_recog = load_model("face_recog.h5")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
count = 0
counter = 0
# FPS update time in seconds
prev_frame_time = 1
# used to record the time at which we processed current frame
new_frame_time = 0

while True:
	frame = vs.read()
	flip_frame = cv2.flip(frame, 1)
	frame = imutils.resize(flip_frame, width=720)

	# (faces, label) = detect_and_predict_face(frame, faceNet, model_recog)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	# blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []

	pred = {"[0]": "edi", "[1]": "unknown"}
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face_detect = frame[startY:endY, startX:endX]
			face_detect = cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB)
			face_detect = cv2.resize(face_detect, (224, 224), interpolation=cv2.INTER_LINEAR)
			# face = img_to_array(face)
			# face = preprocess_input(face)

			face_detect = face_detect / 255
			face_detect = face_detect.reshape(1, 224, 224, 3)
			# face_detect = preprocess_input(face_detect)
			faces.append(face_detect)

			preds = np.argmax(model_recog.predict(face_detect, 1, verbose=0), axis=1)
			facial = pred[str(preds)]

			if facial == "edi":
				label = "Edi"
				color = (0, 255, 0)
			if facial == "unknown":
				label = "Unknown"
				color = (0, 0, 255)

			# print(facial)

			label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
			cv2.putText(frame, label, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	# print(len(faces))

	if len(faces) == 0:
		count = count + 1
		print("[INFO] Found {0} Faces. ".format(len(faces)), count)

	if len(faces) == 1:
		def get_window_hwnd(title):
			for wnd in enum_windows():
				if title.lower() in win32gui.GetWindowText(wnd).lower():
					return wnd
			return 0

		def enum_windows():
			def callback(wnd, data):
				windows.append(wnd)

			windows = []
			win32gui.EnumWindows(callback, None)
			return windows

		window = get_window_hwnd("Python-VEnv")

		tup = win32gui.GetWindowPlacement(window)
		count = 0
		if label == "Edi" and tup[1] == win32con.SW_SHOWMINIMIZED:
			counter = counter + 1
			print("[INFO] Found {0} Faces. ".format(len(faces)), counter)

	if len(faces) >= 2:
		count = count + 1
		print("[INFO] Found {0} Faces. ".format(len(faces)), count)

	if count == 15:
		keyboard = Controller()
		keyboard.press(Key.cmd)
		keyboard.press('d')
		keyboard.release('d')
		keyboard.release(Key.cmd)
		# count = 0
		print("lakukan aksi")

	if counter == 10:
		keyboard = Controller()
		keyboard.press(Key.cmd)
		keyboard.press('d')
		keyboard.release('d')
		keyboard.release(Key.cmd)
		counter = 0

	new_frame_time = time.time()
	fps = 1 / (new_frame_time - prev_frame_time)
	prev_frame_time = new_frame_time
	fps = int(fps)
	fps = str(fps)

	cv2.putText(frame, "FPS : " + fps, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 4, cv2.LINE_AA)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()