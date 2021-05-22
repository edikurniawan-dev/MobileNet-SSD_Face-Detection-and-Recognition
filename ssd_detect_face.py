from imutils.video import VideoStream
import numpy as np
import time
import cv2

prototxt_path = "face-detector-model/ssd_face.prototxt"
weights_path = "face-detector-model/ssd_face.caffemodel"

face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

prev_frame_time = 0
new_frame_time = 0

while True:
	frame = vs.read()
	frame = cv2.flip(frame, 1)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	face_net.setInput(blob)
	detections = face_net.forward()

	faces = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]

			label = str("%.2f" % (confidence * 100))
			label_full = "Face : " + label + "%"
			label_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2), (255, 255, 255), cv2.FILLED)
			cv2.putText(frame, label_full, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	# show the output frame
	new_frame_time = time.time()

	# fps will be number of frame processed in given time frame
	# since their will be most of time error of 0.001 second
	# we will be subtracting it to get more accurate result
	fps = 1 / (new_frame_time - prev_frame_time)
	prev_frame_time = new_frame_time

	# converting the fps into integer
	fps = int(fps)

	# converting the fps to string so that we can display it on frame
	# by using putText function
	fps = "FPS:"+str(fps)
	cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q") or key == ord("Q"):
		break

cv2.destroyAllWindows()
vs.stop()