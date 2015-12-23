import glob
import os
import numpy as np
import math
import cv2
import scipy.io as sio

def c3d_classify(vid_name, mean_file, net, start_frame=0, num_frames=0):

    num = 25
    num_categories = 131

    if num_frames == 0:
        imglist = glob.glob(os.path.join(vid_name, '*image_*.jpg'))
        duration = len(imglist)
    else:
        duration = num_frames

    # selection
    step = int(math.floor((duration-1)/(num-1)))
    dims = (256,340,3,num)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num):
        img_file = os.path.join(vid_name, 'image_{0:05d}.jpg'.format(i*step+1))
        #print "[Info] img_file={}".format(img_file)
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]

    # crop
    rgb_1 = rgb[:224, :224, :,:]
    rgb_2 = rgb[:224, -224:, :,:]
    rgb_3 = rgb[16:240, 60:284, :,:]
    rgb_4 = rgb[-224:, :224, :,:]
    rgb_5 = rgb[-224:, -224:, :,:]
    rgb_f_1 = rgb_flip[:224, :224, :,:]
    rgb_f_2 = rgb_flip[:224, -224:, :,:]
    rgb_f_3 = rgb_flip[16:240, 60:284, :,:]
    rgb_f_4 = rgb_flip[-224:, :224, :,:]
    rgb_f_5 = rgb_flip[-224:, -224:, :,:]

    rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

    # substract mean
    d = sio.loadmat(mean_file)
    image_mean = d['image_mean']
    #print "[Info] rgb.shape={}".format(rgb.shape)
    #print "[Info] image_mean.shape={}".format(image_mean.shape)

    rgb = rgb[:,:,::-1,:] - np.tile(image_mean[...,np.newaxis], (1, 1, 1, rgb.shape[3]))
    rgb = np.transpose(rgb, (1,0,2,3))

    # test
    batch_size = 50
    prediction = np.zeros((num_categories,rgb.shape[3]))
    num_batches = int(math.ceil(float(rgb.shape[3])/batch_size))

    for bb in range(num_batches):
        span = range(batch_size*bb, min(rgb.shape[3],batch_size*(bb+1)))
        #print "span={}, len(span)={}".format(span, len(span))

        #print "input={}".format(np.transpose(rgb[:,:,:,span], (3,2,1,0)))
        net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,span], (3,2,1,0))
        output = net.forward()
        #print "output={}".format(np.transpose(output['fc8-2']))

        prediction[:, span] = np.transpose(output['fc8-2'])

    return prediction
