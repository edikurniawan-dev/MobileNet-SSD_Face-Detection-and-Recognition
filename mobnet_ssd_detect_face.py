import os
import cv2
import numpy as np

# Define paths
# base_dir = os.path.dirname(__file__)
# prototxt_path = 'face_detector/deploy_mask.prototxt'
# caffemodel_path = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

prototxt_path = 'MobilenetSSDFace-master/models/deploy/ssd-face.prototxt'
caffemodel_path = 'MobilenetSSDFace-master/models/deploy/ssd-face.caffemodel'

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# load all image in folder
# for file in os.listdir(base_dir + 'images'):
#     file_name, file_extension = os.path.splitext(file)
#     if (file_extension in ['.png','.jpg']):
#         print("Image path: {}".format(base_dir + 'images/' + file))


image = cv2.imread("elif.jpg")
(h, w) = image.shape[:2]
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

model.setInput(blob)
detections = model.forward()

# Create frame around face
for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    confidence = detections[0, 0, i, 2]

  # If confidence > 0.5, show box around face
    if (confidence > 0.5):
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
        frame = image[startY:endY, startX:endX]
        cv2.imwrite('face' + str(i) + '.jpg', frame)
        # cv2.imwrite('image.jpg', image)

print("Image converted successfully")

# Identify each face
# for i in range(0, detections.shape[2]):
#
#
#     if (confidence > 0.5):
        # count += 1
