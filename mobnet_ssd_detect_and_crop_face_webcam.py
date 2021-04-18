import os
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

def detect_and_crop_face(frame, key):
    (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            frame = frame[startY:endY, startX:endX]

            if key == 32:
                cv2.imwrite('dataset/testing/face' + str(count) + '.jpg', frame)
                print("Image cropped successfully")

# prototxt_path = 'face-detector-model/ssd-model/deploy.prototxt'
# caffemodel_path = 'face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel'

prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
caffemodel_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

vs = VideoStream(src=0).start()
count = 0
while True:
    frame = vs.read()
    flip_frame = cv2.flip(frame, 1)
    frame = imutils.resize(flip_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

    if key == 32:
        detect_and_crop_face(frame, key)
        count += 1
    else:
        detect_and_crop_face(frame, key=0)

    cv2.imshow("Build Dataset", frame)

cv2.destroyAllWindows()
vs.stop()
