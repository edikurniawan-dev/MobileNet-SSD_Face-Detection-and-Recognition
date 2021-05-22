from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import time
from pynput.keyboard import Key, Controller
import cv2


prototxt_path = "face-detector-model/ssd_face.prototxt"
weights_path = "face-detector-model/ssd_face.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

model_recog = load_model("dataset/result/result20/face_recog20.h5")

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
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []

    pred = {"0", "1"}
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        if confidence > 0.5:
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY+60, startX:endX+95]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)

            preds = model_recog.predict(face)
            label_percent = str("{:.2f}%".format((max(max(preds))) * 100))

            preds = np.argmax(preds)

            if preds == 0:
                label = "Edi"
                color = (0, 255, 0)
            if preds == 1:
                label = "Unknown"
                color = (0, 0, 255)

            label_full = label + " " + label_percent
            label_size = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.rectangle(frame, (startX, startY - 20), ((startX + label_size[0][0]) + 10, startY - 2),(255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_full, (startX + 4, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

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

    cv2.putText(frame, "FPS: " + fps, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Privacy Guard", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()