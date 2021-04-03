import caffe

try:
    caffe.Net('mobnet-ssd-model/ssd_face/ssd_face_train.prototxt',
              'mobnet-ssd-model/empty.caffemodel',
              caffe.TRAIN)
    caffe.Net('mobnet-ssd-model/ssd_face/ssd_face_test.prototxt',
              'mobnet-ssd-model/empty.caffemodel',
              caffe.TEST)
    caffe.Net('mobnet-ssd-model/ssd_face/ssd_face_deploy.prototxt',
              'mobnet-ssd-model/empty.caffemodel',
              caffe.TEST)
    caffe.Net('mobnet-ssd-model/ssd_face/ssd_face_deploy_bn.prototxt',
              'mobnet-ssd-model/empty.caffemodel',
              caffe.TEST)
    print('Model check COMPLETE')
except Exception as e:
    print(repr(e))
    print('Model check FAILED')
    