import cv2 as cv
from cv2 import dnn
import time
from imutils.video import VideoStream

inWidth = 300
inHeight = 300
confThreshold = 0.5

# prototxt = r'face-detector-model/deploy.prototxt'
# caffemodel = r'face-detector-model/res10_300x300_ssd_iter_140000.caffemodel'
# prototxt = 'MobilenetSSDFace-master/mobnet-ssd-model/other-model-face-detector/ssd-face.prototxt'
# caffemodel = 'MobilenetSSDFace-master/mobnet-ssd-model/other-model-face-detector/ssd-face.caffemodel'

prototxt = 'face-detector-model/mobnet-ssd-model/deploy/ssd-face.prototxt'
caffemodel = 'face-detector-model/mobnet-ssd-model/deploy/ssd-face.caffemodel'



net = dnn.readNetFromCaffe(prototxt, caffemodel)

cap = cv.VideoCapture(0)
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    # vs = VideoStream(src=0).start()
    ret, frame = cap.read()
    # flip_frame = cv.flip(frame, 1)
    cols = frame.shape[1]
    rows = frame.shape[0]


    # frame = imutils.resize(flip_frame, width=540)
    # net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
    net.setInput(cv.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    detections = net.forward()

    perf_stats = net.getPerfProfile()

    print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

    new_frame_time = time.time()

    # Calculating the fpsq

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = "FPS : "+str(fps)

    cv.putText(frame, fps, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv.LINE_AA)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confThreshold:
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # xLeftBottom = int(detections[0, 0, i, 3])
            # yLeftBottom = int(detections[0, 0, i, 4])
            # xRightTop = int(detections[0, 0, i, 5])
            # yRightTop = int(detections[0, 0, i, 6])

            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
            label = "face: %.4f" % confidence
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow("detections", frame)

    key = cv.waitKey(1) & 0xFF
    total = 0
    if key == 32:
        print("Image captured!")
        # p = os.path.sep.join([args["output"], "{}.png".format(
        # 	str(total).zfill(5))])
        cv.imwrite("{}.png".format(str(total).zfill(5)))
        total += 1

    if cv.waitKey(1) != -1:
        break