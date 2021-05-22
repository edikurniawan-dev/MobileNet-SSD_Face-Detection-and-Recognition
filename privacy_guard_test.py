from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2
import time
from pynput.keyboard import Key, Controller
# import caffe

prototxt_path = "face-detector-model/ssd_face.prototxt"
weights_path = "face-detector-model/ssd_face.caffemodel"


faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

model_recog = load_model("face_recog_good.h5")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
counter_1 = 0
counter_2 = 0
counter_3 = 0
prev_frame_time = 1
new_frame_time = 0
window = "show"

def windows_d():
	keyboard = Controller()
	keyboard.press(Key.cmd)
	keyboard.press('d')
	keyboard.release('d')
	keyboard.release(Key.cmd)

while True:
	frame = vs.read()
	frame = cv2.flip(frame, 1)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []

	pred = {"[0]": "edi", "[1]": "unknown"}
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# face_detect = frame[startY:endY, startX:endX]
			face_detect = frame[startY:endY+60, startX:endX+95]
			# face_detect = frame[startY - 80: endY + 60, startX - 95: endX + 95]

			face_detect = cv2.resize(face_detect, (224, 224), interpolation=cv2.INTER_LINEAR)
			face_detect = face_detect / 255
			face_detect = face_detect.reshape(1, 224, 224, 3)
			faces.append(face_detect)

			preds = model_recog.predict(face_detect)
			label_percent = str("{:.2f}%".format((max(max(preds)))*100))

			preds = np.argmax(preds, axis=1)
			facial = pred[str(preds)]

			if facial == "edi":
				label = "Edi"
				color = (0, 255, 0)
			if facial == "unknown":
				label = "Unknown"
				color = (0, 0, 255)

			# print(facial)
			label_full = label+" "+label_percent
			label_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
			cv2.putText(frame, label_full, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	# print(len(faces))
	#
	if len(faces) == 0 and window == "show":
		counter_3 = 0
		counter_2 = 0
		counter_1 += 1
		print("[INFO] Found {0} Faces. ".format(len(faces)), counter_1, window, "kondisi 1")
		if counter_1 == 15:
			windows_d()
			window = "hide"
			print("[INFO] Found {0} Faces. ".format(len(faces)), counter_1, window, "kondisi 1")

	if len(faces) == 1 and label == "Edi":
		counter_3 = 0
		counter_1 = 0
		if window == "hide":
			counter_2 += 1
			# print(len(faces))
			print("[INFO] Found {0} Faces. ".format(len(faces)), counter_2, window, "kondisi 2")

			if counter_2 == 10:
				windows_d()
				window = "show"
				print("[INFO] Found {0} Faces. ".format(len(faces)), counter_2, window, "kondisi 2")

	if len(faces) == 1 and label == "Unknown":
		counter_1 = 0
		counter_2 = 0

		if window == "show":
			counter_3 += 1
			# print(len(faces))
			print("[INFO] Found {0} Faces. ".format(len(faces)), counter_3, window, "kondisi 3")

			if counter_3 == 10:
				windows_d()
				window = "hide"
				print("[INFO] Found {0} Faces. ".format(len(faces)), counter_3, window, "kondisi 3")

	if len(faces) >= 2:
		counter_1 += 1
		if counter_1 == 5:
			windows_d();
			window = "hide"
			# counter_2 = 0
			print("[INFO] Found {0} Faces. ".format(len(faces)), counter_2, window)

	new_frame_time = time.time()
	fps = 1 / (new_frame_time - prev_frame_time)
	prev_frame_time = new_frame_time
	fps = int(fps)
	fps = str(fps)

	cv2.putText(frame, "FPS : " + fps, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 4, cv2.LINE_AA)
	cv2.imshow("Privacy Guard", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()