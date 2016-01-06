import os
import numpy as np
import math
import cv2

def c3d_classify(vid_name, image_mean, net, start_frame):
    ''' start_frame is 0-based whereas the first image file is image_00001.jpg '''

    c3d_depth = 16 # num of frames
    num_categories = 131
    augment_input = False

    # selection
    dims = (128,171,3,c3d_depth)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(c3d_depth):
        img_file = os.path.join(vid_name, 'image_{0:05d}.jpg'.format(start_frame+i+1))
        print "[info] vid_name={}, start_frame={}, img_file={}".format(vid_name, start_frame, img_file)
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]

    # substract mean
    image_mean = np.transpose(image_mean, (2,3,1,0))
    rgb -= image_mean
    rgb_flip -= image_mean[:,::-1,:,:]

    if augment_input:
        # crop (112-by-112)
        rgb_1 = rgb[:112, :112, :,:]
        rgb_2 = rgb[:112, -112:, :,:]
        rgb_3 = rgb[8:120, 30:142, :,:]
        rgb_4 = rgb[-112:, :112, :,:]
        rgb_5 = rgb[-112:, -112:, :,:]
        rgb_f_1 = rgb_flip[:112, :112, :,:]
        rgb_f_2 = rgb_flip[:112, -112:, :,:]
        rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        rgb_f_4 = rgb_flip[-112:, :112, :,:]
        rgb_f_5 = rgb_flip[-112:, -112:, :,:]

        rgb = np.concatenate((rgb_1[...,np.newaxis],
                              rgb_2[...,np.newaxis],
                              rgb_3[...,np.newaxis],
                              rgb_4[...,np.newaxis],
                              rgb_5[...,np.newaxis],
                              rgb_f_1[...,np.newaxis],
                              rgb_f_2[...,np.newaxis],
                              rgb_f_3[...,np.newaxis],
                              rgb_f_4[...,np.newaxis],
                              rgb_f_5[...,np.newaxis]), axis=4)
    else:
        rgb_3 = rgb[8:120, 30:142, :,:]
        rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        rgb = np.concatenate((rgb_3[...,np.newaxis],
                              rgb_f_3[...,np.newaxis]), axis=4)

    #rgb = np.transpose(rgb, (1,0,2,3))
    #print "net.blobs['data']={}".format(net.blobs['data'])
    #print "net.blobs['data'].data={}".format(net.blobs['data'].data)
    #print "net.blobs['data'].data.shape={}".format(net.blobs['data'].data.shape)

    #test_blob = np.zeros(shape=(2,3,16,112,112), dtype=np.float64)
    #net.blobs['data'].data[...] = test_blob
    #output = net.forward()

    batch_size = 2
    prediction = np.zeros((num_categories,rgb.shape[4]))
    num_batches = int(math.ceil(float(rgb.shape[4])/batch_size))
    #print "prediction.shape={}, rgb.shape={}, num_batches={}".format(prediction.shape, rgb.shape, num_batches)

    for bb in range(num_batches):
        span = range(batch_size*bb, min(rgb.shape[4],batch_size*(bb+1)))
        #print "span={}, len(span)={}".format(span, len(span))

        net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,:,span], (4,2,3,0,1))
        output = net.forward()
        #print "output['fc8'].shape={}".format(output['fc8'].shape)

        prediction[:, span] = np.transpose(np.squeeze(output['fc8']))

    return prediction

'''
layers {
  name: "data"
  type: VIDEO_DATA
  top: "data"
  top: "label"
  image_data_param {
    source: "dextro_benchmark_val_flow_smaller.txt"
    use_image: false
    mean_file: "train01_16_128_171_mean.binaryproto"
    use_temporal_jitter: false
    #batch_size: 30
    #batch_size: 2
    batch_size: 2
    crop_size: 112
    mirror: false
    show_data: 0
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: true
  }
}
'''
