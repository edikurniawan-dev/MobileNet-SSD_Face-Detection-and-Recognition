import cv2 as cv
from cv2 import dnn
import time
from imutils.video import FPS


inWidth = 300
inHeight = 300
confThreshold = 0.5

prototxt = r'face_detector/deploy_mask.prototxt'
caffemodel = r'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
# prototxt = 'MobilenetSSDFace-master/models/deploy/ssd-face.prototxt'
# caffemodel = 'MobilenetSSDFace-master/models/deploy/ssd-face.caffemodel'

# prototxt = 'MobilenetSSDFace-master/models/deploy/ssd-face-longrange.prototxt'
# caffemodel = 'MobilenetSSDFace-master/models/deploy/ssd-face-longrange.caffemodel'

# prototxt = 'MobilenetSSDFace-master/models/ssd_face_pruned/face_deploy.prototxt'
# caffemodel = 'MobilenetSSDFace-master/models/ssd_face_pruned/short_init.caffemodel'
if __name__ == '__main__':
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    # net = dnn.readNetFromTensorflow(pb, pbt)  # mobnet-ssd

    cap = cv.VideoCapture(0)
    # cap = cv.VideoStream(src=0).start()
    # fps = FPS().start()
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        flip_frame = cv.flip(frame, 1)
        cols = flip_frame.shape[1]
        rows = flip_frame.shape[0]

        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        # detections = net.forward()

        # net.setInput(cv.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        detections = net.forward()

        perf_stats = net.getPerfProfile()

        print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 2, cv.LINE_AA)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)


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
            cv.imwrite("{}.png".format(str(total).zfill(5)), orig)
            total += 1

        if cv.waitKey(1) != -1:
            break