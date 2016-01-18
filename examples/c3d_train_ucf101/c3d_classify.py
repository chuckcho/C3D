import os
import numpy as np
import math
import cv2

def c3d_classify(
        vid_name,
        image_mean,
        net,
        start_frame,
        prob_layer='prob',
        multi_crop=False
        ):
    ''' start_frame is 1-based and the first image file is image_0001.jpg '''

    #print "net.blobs={}".format(
    #    net.blobs
    #)
    #print "net.blobs['prob'].data.shape={}".format(
    #    net.blobs['prob'].data.shape
    #)
    #print "net.blobs['data'].data.shape={}".format(
    #    net.blobs['data'].data.shape
    #)

    # infer net params
    batch_size = net.blobs['data'].data.shape[0]
    c3d_depth = net.blobs['data'].data.shape[2]
    num_categories = net.blobs['prob'].data.shape[1]

    # selection
    dims = (128,171,3,c3d_depth)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(c3d_depth):
        #img_file = os.path.join(vid_name, 'image_{0:05d}.jpg'.format(start_frame+i+1))
        img_file = os.path.join(vid_name, 'image_{0:04d}.jpg'.format(start_frame+i))
        #print "[info] vid_name={}, start_frame={}, img_file={}".format(vid_name, start_frame, img_file)
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        #print "[info] img.shape={}".format(img.shape)
        img = cv2.resize(img, dims[1::-1])
        #print "[info] img.shape={}".format(img.shape)
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]

    # substract mean
    #print "image_mean.shape={}".format(image_mean.shape)
    image_mean = np.transpose(np.squeeze(image_mean), (2,3,0,1))
    rgb -= image_mean
    rgb_flip -= image_mean[:,::-1,:,:]

    if multi_crop:
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
        #rgb = rgb_3[...,np.newaxis]

    #rgb = np.transpose(rgb, (1,0,2,3))
    #print "net.blobs['data']={}".format(net.blobs['data'])
    #print "net.blobs['data'].data={}".format(net.blobs['data'].data)
    #print "net.blobs['data'].data.shape={}".format(net.blobs['data'].data.shape)

    #test_blob = np.zeros(shape=(2,3,16,112,112), dtype=np.float64)
    #net.blobs['data'].data[...] = test_blob
    #output = net.forward()

    prediction = np.zeros((num_categories,rgb.shape[4]))

    if rgb.shape[4] < batch_size:
        print "rgb.shape[4]={}, batch_size={}".format(rgb.shape[4], batch_size)
        net.blobs['data'].data[:rgb.shape[4],:,:,:,:] = np.transpose(rgb, (4,2,3,0,1))
        net.blobs['data'].data[rgb.shape[4]:,:,:,:,:] = np.zeros(
                (batch_size-rgb.shape[4], rgb.shape[2], rgb.shape[3], rgb.shape[0], rgb.shape[1])
                )
        output = net.forward()
        #print "output={}".format(output)
        #print "output[prob_layer].shape={}".format(output[prob_layer].shape)
        #print "np.transpose(np.squeeze(output[prob_layer][:rgb.shape[4],:,:,:,:])).shape={}".format(
        #        np.transpose(np.squeeze(output[prob_layer][:rgb.shape[4],:,:,:,:])).shape )
        prediction = np.transpose(np.squeeze(output[prob_layer][:rgb.shape[4],:,:,:,:]))
    else:
        num_batches = int(math.ceil(float(rgb.shape[4])/batch_size))
        #print "prediction.shape={}, rgb.shape={}, num_batches={}".format(prediction.shape, rgb.shape, num_batches)
        for bb in range(num_batches):
            span = range(batch_size*bb, min(rgb.shape[4],batch_size*(bb+1)))
            net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,:,span], (4,2,3,0,1))
            output = net.forward()
            #print "output[prob_layer].shape={}".format(
            #        output[prob_layer].shape
            #        )
            prediction[:, span] = np.transpose(np.squeeze(output[prob_layer], axis=(2,3,4)))

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
    batch_size: 50
    crop_size: 112
    mirror: false
    show_data: 0
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: false
  }
}
'''
