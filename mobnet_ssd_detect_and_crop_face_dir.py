import os
import cv2
import numpy as np

def detect_and_crop_face(image, count):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    # print(count)
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            # cv2.rectangle(path_image, (startX, startY), (endX, endY), (255, 255, 255), 2)
            frame_crop = image[startY:endY, startX:endX]
            frame_crop = cv2.resize(frame_crop, (224, 224))
            cv2.imwrite('dataset/edi_val_crop/' + str(count) + '.jpg', frame_crop)
            print("success")
            count += 1
    return(count)

prototxt_path = 'face-detector-model/ssd-model/deploy.prototxt'
caffemodel_path = 'face-detector-model/ssd-model/res10_300x300_ssd_iter_140000.caffemodel'

# prototxt_path = 'face-detector-model/mobnet-ssd-model/ssd-face.prototxt'
# caffemodel_path = 'face-detector-model/mobnet-ssd-model/ssd-face.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
path = "dataset/edi_val/"
count=0

many_file_in_dict = len(os.listdir(path))
filename = os.listdir(path)

for i in range(many_file_in_dict):
    file_image = os.path.join(path+filename[i])
    image = cv2.imread(file_image)
    count = detect_and_crop_face(image, count)
