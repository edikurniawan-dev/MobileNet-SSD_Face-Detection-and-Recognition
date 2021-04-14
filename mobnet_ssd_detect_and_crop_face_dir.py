import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import imutils
# from imutils.video import VideoStream
import random

def detect_and_crop_face(path_image, count):
    frame = cv2.imread(path_image)
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
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            cv2.imwrite('dataset/habib_crop/face' + str(count) + '.jpg', frame)
            frame = frame[startY:endY, startX:endX]

    # cv2.imwrite('dataset/habib_crop/face' + str(count) + '.jpg', frame)
    # print("Image cropped successfully")
    # os.remove("dataset/habib_crop/image_capture.jpg")

# prototxt_path = 'face-detector-model/ssd-model/deploy.prototxt'
# caffemodel_path = 'face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel'

prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
caffemodel_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# path = os.path.join(path+filename)

path = "dataset/habib/"
many_file_in_dict = len(os.listdir(path))
filename = os.listdir(path)

for i in range(many_file_in_dict):
    file_image = os.path.join(path+filename[i])
    detect_and_crop_face(file_image, i+1)
    # print(file_image)

# count = 0
# print(many_file_in_dict)

# for i in range(many_file_in_dict):
#     path_image = get_random_image(path)
#     print(path_image)

    # input_original = path_image.copy()
    # input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    #
    # input_im = cv2.resize(path_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # input_im = input_im / 255.
    # input_im = input_im.reshape(1, 224, 224, 3)

    # detect_and_crop_face(path_image, i)



# while True:
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels
#     frame = vs.read()
#     flip_frame = cv2.flip(frame, 1)
#     frame = imutils.resize(flip_frame)
#
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `spacebar` key was pressed, write the *original* frame to disk
#     # so we can later process it and use it for face recognition
#     if key == 32:
#         count += 1
#         cv2.imwrite('dataset/edi/image_capture.jpg', flip_frame)
#         print("Image captured!")
#         detect_and_crop_face(frame, key)
#     else:
#         detect_and_crop_face(frame, key=0)
#
#     cv2.imshow("Build Dataset", frame)
#
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q") or key == ord("Q"):
#         break
#
# cv2.destroyAllWindows()
# vs.stop()
