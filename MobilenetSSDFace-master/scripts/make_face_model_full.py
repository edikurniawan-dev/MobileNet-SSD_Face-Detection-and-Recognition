import caffe

ref_net = caffe.Net('mobnet-ssd-model/ssd_voc/deploy_bn.prototxt',
                    'mobnet-ssd-model/ssd_voc/mobilenet_iter_73000.caffemodel',
                    caffe.TRAIN)
new_net = caffe.Net('mobnet-ssd-model/ssd_face/ssd_face_deploy_bn.prototxt',
                    'mobnet-ssd-model/empty.caffemodel',
                    caffe.TRAIN)
for pn,blobs in new_net.params.items():
    print(pn)
    for i in range(len(blobs)):
        print(blobs[i].data.shape)
        
for ln in new_net.params.keys():
    print('Layer '+ln)
    skip = True
    for i in range(min([len(ref_net.params[ln]), len(new_net.params[ln])])):
        if new_net.params[ln][i].data.shape == ref_net.params[ln][i].data.shape:
            new_net.params[ln][i].data[...] = ref_net.params[ln][i].data
            print('\tParameter '+str(i))
            skip = False
    if skip:
        print('Skipping')

new_net.save('mobnet-ssd-model/ssd_face/face_init_full.caffemodel')

print('Model complete')