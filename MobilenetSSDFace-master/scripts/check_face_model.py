import caffe

try:
    caffe.Net('mobnet-ssd-model/ssd_face_pruned/face_train.prototxt',
              'mobnet-ssd-model/ssd_face_pruned/face_init.caffemodel',
              caffe.TRAIN)
    caffe.Net('mobnet-ssd-model/ssd_face_pruned/face_test.prototxt',
              'mobnet-ssd-model/ssd_face_pruned/face_init.caffemodel',
              caffe.TEST)
    print('Model check COMPLETE')
except Exception as e:
    print(repr(e))
    print('Model check FAILED')