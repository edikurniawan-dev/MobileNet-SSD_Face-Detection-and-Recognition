import os
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

prototxt_path = 'face-detector-model/ssd-model/deploy.prototxt'
caffemodel_path = 'face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel'

# prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
# caffemodel_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# load all image in folder
# for file in os.listdir(base_dir + 'images'):
#     file_name, file_extension = os.path.splitext(file)
#     if (file_extension in ['.png','.jpg']):
#         print("Image path: {}".format(base_dir + 'images/' + file))

vs = VideoStream(src=0).start()

def detect_face(frame, key):
    if key == 32:
        frame = cv2.imread("dataset/edi/image_capture.jpg")
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            # cv2.imwrite('dataset/edi/image_face_box.jpg', image)
            frame = frame[startY:endY, startX:endX]

    if (key == 32):
        cv2.imwrite('dataset/edi/face' + str(count) + '.jpg', frame)
        print("Image cropped successfully")
        os.remove("dataset/edi/image_capture.jpg")

count = 0
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    flip_frame = cv2.flip(frame, 1)
    frame = imutils.resize(flip_frame, width=540)

    key = cv2.waitKey(1) & 0xFF

    # if the `spacebar` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == 32:
        count += 1
        cv2.imwrite('dataset/edi/image_capture.jpg', flip_frame)
        print("Image captured!")
        # crop_face(count)
        detect_face(frame, key)
    else:
        detect_face(frame, key=0)

    cv2.imshow("detections", frame)

    # if the `q` key was pressed, break from the loop
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
vs.stop()
