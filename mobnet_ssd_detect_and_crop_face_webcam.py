import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

def detect_and_crop_face(frame, key):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)

    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        # print(i)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        if (confidence > 0.5):
            if key == 0:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            if key == 32:
                frame = frame[startY-80:endY+60, startX-95:endX+95]
                cv2.imwrite('dataset/' + str(count) + '.jpg', frame)
                print("Image cropped successfully")

prototxt_path = "face-detector-model/ssd_face.prototxt"
caffemodel_path = "face-detector-model/ssd_face.caffemodel"

pb = 'face-detector-model/frozen_inference_graph_face.pb'
pbt = 'face-detector-model/face_label_map.pbtxt'
Net_SSD = cv2.dnn.readNetFromTensorflow(pb, pbt)

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

vs = VideoStream(src=0).start()
count = 0
while True:
    frame = vs.read()
    flip_frame = cv2.flip(frame, 1)
    frame = imutils.resize(flip_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        detect_and_crop_face(frame, key)
        count += 1
    else:
        detect_and_crop_face(frame, key=0)

    cv2.imshow("Build Dataset", frame)

    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
vs.stop()
