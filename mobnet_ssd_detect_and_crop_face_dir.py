import os
# from os import listdir
# from os.path import isfile, join
import cv2
import numpy as np

def detect_and_crop_face(path_image, count):
    frame = cv2.imread(path_image)
    (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    model.setInput(blob)
    detections = model.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            frame_crop = frame[startY:endY, startX:endX]
            cv2.imwrite('dataset/2face' + str(i) + '.jpg', frame_crop)
            print(i)
            print("ada")
            faces.append(frame)

    print(len(faces))

    # print(len(faces))
    # cv2.imwrite('dataset/habib_crop/face' + str(count) + '.jpg', frame)
    # cv2.imwrite('dataset/unknown_crop/face' + str(count) + '.jpg', frame)
    # print("Image cropped successfully")

# prototxt_path = 'face-detector-model/ssd-model/deploy.prototxt'
# caffemodel_path = 'face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel'

prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
caffemodel_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# path = "dataset/habib/"
path = "dataset/2face.jpg"
i=0
detect_and_crop_face(path, i)

# many_file_in_dict = len(os.listdir(path))
# filename = os.listdir(path)
#
# for i in range(many_file_in_dict):
#     file_image = os.path.join(path+filename[i])
#     detect_and_crop_face(file_image, i+1)
