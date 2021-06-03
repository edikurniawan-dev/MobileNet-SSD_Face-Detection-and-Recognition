from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

base_model = MobileNetV2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))

head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)
head_model = Dense(128,activation='relu')(head_model)
head_model = Dense(2, activation='softmax')(head_model)

model_train = Model(inputs = base_model.input, outputs = head_model)

for layer in base_model.layers:
	layer.trainable = False

model_train.summary()

directory_dataset = "/content/drive/MyDrive/FaceRecog/new_dataset/dataset20"
category_dataset = ["edi", "unknown"]

data = []
labels = []

for category in category_dataset:
	path = os.path.join(directory_dataset, category)
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		image = load_img(img_path, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)
		data.append(image)
		labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)
# print(labels)


learning_rate_size = 1e-4
epoch_size = 10
batch_size_number = 8

(trainX, testX, trainY, testY) = train_test_split(data,
												  labels,
												  test_size=0.20,
												  stratify=labels,
												  random_state=42)

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

checkpoint = ModelCheckpoint("/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/face_recog20.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [earlystop, checkpoint]

print("[INFO] compiling model...")
opt = Adam(lr=learning_rate_size, decay=learning_rate_size / epoch_size)
model_train.compile(loss="binary_crossentropy",
                    optimizer=opt,
                    metrics=["accuracy"])

print("[INFO] training head...")
H = model_train.fit(aug.flow(trainX, trainY, batch_size=batch_size_number),
                    steps_per_epoch=len(trainX) // batch_size_number,
                    validation_data=(testX, testY),
                    validation_steps=len(testX) // batch_size_number,
                    epochs=epoch_size,
                    callbacks = callbacks)

predIdxs = model_train.predict(testX, batch_size=batch_size_number)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

plt.style.use("ggplot")
plt.figure()

plt.plot(H.history['accuracy'] , label = 'train acc')
plt.plot(H.history['val_accuracy'] , label = 'val acc')
plt.plot(H.history['loss'] , label = 'train loss')
plt.plot(H.history['val_loss'] , label = 'val loss')

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/result20.png")

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow
import numpy as np
import imutils
import time
import cv2
import os
from os import listdir
from os.path import isfile, join

prototxt_path = "/content/drive/MyDrive/FaceRecog/ssd_face.prototxt"
weights_path = "/content/drive/MyDrive/FaceRecog/ssd_face.caffemodel"

faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

# load the face mask detector model from disk
model_recog = load_model('/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/face_recog20.h5')

path = "/content/drive/MyDrive/FaceRecog/new_dataset/test/edi/"
# path = "/content/drive/MyDrive/FaceRecog/new_dataset/test/unknown/"

many_file_in_dict = len(os.listdir(path))
image_file = os.listdir(path)
# print(image_file)

for count in range(many_file_in_dict):
	path_image = os.path.join(path + image_file[count])
	input_image = cv2.imread(path_image)

	# cv2_imshow(input_image)

	(h, w) = input_image.shape[:2]
	blob = cv2.dnn.blobFromImage(input_image, 1.0, (224, 224), (104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	# faces = []
	pred = {"0", "1"}

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face_detect = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
			face_detect = cv2.resize(face_detect, (224, 224))
			face_detect = img_to_array(face_detect)
			face_detect = preprocess_input(face_detect)
			face_detect = np.expand_dims(face_detect, axis=0)

			preds = model_recog.predict(face_detect)
			label_percent = str("{:.2f}%".format((max(max(preds))) * 100))

			preds = np.argmax(preds)

			# facial = pred[str(preds)]

			if preds == 0:
				label = "Edi"
				color = (0, 255, 0)
			if preds == 1:
				label = "Unknown"
				color = (0, 0, 255)

			label_full = label + " " + label_percent
			label_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(input_image, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(input_image, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2),
						  (255, 255, 255), cv2.FILLED)
			cv2.putText(input_image, label_full, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	image_resize = cv2.resize(input_image, (480, 480))
	# print(preds)
	cv2_imshow(image_resize)
	cv2.imwrite("/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/test" + str(count + 1) + "-20.jpg",
				image_resize)
	time.sleep(1)
	print('\n')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow
import numpy as np
import imutils
import time
import cv2
import os
from os import listdir
from os.path import isfile, join

prototxt_path = "/content/drive/MyDrive/FaceRecog/ssd_face.prototxt"
weights_path = "/content/drive/MyDrive/FaceRecog/ssd_face.caffemodel"

faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

# load the face mask detector model from disk
model_recog = load_model('/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/face_recog20.h5')

# path = "/content/drive/MyDrive/FaceRecog/new_dataset/test/edi/"
path = "/content/drive/MyDrive/FaceRecog/new_dataset/test/unknown/"

many_file_in_dict = len(os.listdir(path))
image_file = os.listdir(path)
# print(image_file)

for count in range(many_file_in_dict):
	path_image = os.path.join(path + image_file[count])
	input_image = cv2.imread(path_image)

	# cv2_imshow(input_image)

	(h, w) = input_image.shape[:2]
	blob = cv2.dnn.blobFromImage(input_image, 1.0, (224, 224), (104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	# faces = []
	pred = {"0", "1"}

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		if confidence > 0.5:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face_detect = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
			face_detect = cv2.resize(face_detect, (224, 224))
			face_detect = img_to_array(face_detect)
			face_detect = preprocess_input(face_detect)
			face_detect = np.expand_dims(face_detect, axis=0)

			preds = model_recog.predict(face_detect)
			label_percent = str("{:.2f}%".format((max(max(preds))) * 100))

			preds = np.argmax(preds)

			# facial = pred[str(preds)]

			if preds == 0:
				label = "Edi"
				color = (0, 255, 0)
			if preds == 1:
				label = "Unknown"
				color = (0, 0, 255)

			label_full = label + " " + label_percent
			label_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(input_image, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(input_image, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2),
						  (255, 255, 255), cv2.FILLED)
			cv2.putText(input_image, label_full, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

	image_resize = cv2.resize(input_image, (480, 480))
	# print(preds)
	cv2_imshow(image_resize)
	cv2.imwrite("/content/drive/MyDrive/FaceRecog/new_dataset/result/result20/test" + str(count + 21) + "-20.jpg",
				image_resize)
	time.sleep(1)
	print('\n')