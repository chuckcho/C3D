#!/usr/bin/env python

import sys
sys.path.append("/home/chuck/projects/C3D/python")
import c3d_caffe
import numpy as np
import os

protomean = 'ucf101_train_mean.binaryproto'
npymean = 'ucf101_train_mean.npy'

blob = c3d_caffe.proto.caffe_pb2.BlobProto()
data = open( protomean, 'rb' ).read()
blob.ParseFromString(data)
blob.num = 1
blob.channels = 3
blob.length = 16
blob.height = 128
blob.width = 171
arr = np.array( c3d_caffe.io.blobproto_to_array(blob) )
print "arr.shape={}".format(arr.shape)
#out = arr[0]
#np.save( npymean , out )
np.save( npymean, arr )
