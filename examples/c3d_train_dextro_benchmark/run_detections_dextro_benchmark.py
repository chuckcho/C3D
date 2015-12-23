#!/usr/bin/env python

import os
import numpy as np
import math
import operator
import random
import json

import sys
sys.path.append("/home/chuck/projects/C3D/python")
try:
    import caffe
except ImportError:
    print 'ImportError'
    import caffe

from c3d_classify import c3d_classify

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def get_content_type():
    mapping_file = 'dextro_benchmark_content_type_mapping.json'
    with open(mapping_file, 'r') as fp:
        content_type_mapping = json.load(fp)

    mapping_file = 'dextro_benchmark_content_type_id_mapping.json'
    with open(mapping_file, 'r') as fp:
        content_type_id_mapping = json.load(fp)

    return content_type_mapping, content_type_id_mapping

def main():

    # force save
    force_save = False

    modulenames = set(sys.modules)&set(globals())
    allmodules = [sys.modules[name] for name in modulenames]

    print "allmodules=".format(allmodules)

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # model
    model_def_file = 'c3d_ucf101_finetuning_test.prototxt'
    model_file = 'c3d_dextro_benchmark_iter_5000'
    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # read test video list
    test_video_list = 'dextro_benchmark_val_flow_smaller.txt'
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")

    # top_N
    top_N = 5
    #top_N = 131 # for debugging

    content_type_mapping, content_type_id_mapping = get_content_type()

    for count, video_and_category in enumerate(reader):
        (video_name, start_frame, num_frames, category) = video_and_category
        start_frame = int(start_frame)
        num_frames = int(num_frames)
        category = int(category)
        if not os.path.isdir(video_name):
            print "[Error] video_name path={} does not exist. Skipping...".format(video_name)
            continue
        video_id = video_name.split('/')[-1]
        print "-"*79
        print "video_name={} ({}-th), start_frame={}, num_frames={}, category={}".format(video_name, count+1, start_frame, num_frames, category)

        result = '{0}_frame_{1:05d}_numframe_{2:05d}_category_{3:04d}_rgb.txt'.format(video_id, start_frame, num_frames, category)
        if os.path.isfile(result) and not force_save:
            print "[Info] intermediate output file={} for RGB has been already saved. Skipping...".format(result)
            avg_pred = np.loadtxt(result)
        else:
            mean_file = 'mean.mat'
            prediction = VideoSpatialPrediction(video_name, mean_file, net, start_frame, num_frames)
            avg_pred_fc8 = np.mean(prediction, axis=1)
            avg_pred = softmax(avg_pred_fc8)
            np.savetxt(result, avg_pred, delimiter=",")
        sorted_indices = sorted(range(len(avg_pred)), key=lambda k: avg_pred[k])
        print "-"*5
        for x in range(top_N):
            index = sorted_indices[-x-1]
            prob = round(avg_pred[index]*100,10)
            if category == index:
                hit_or_miss = '!!!!!!!!!!!!!!!  hit !!!!!!!!!!!!!!!'
            else:
                hit_or_miss = ''
            print "[Info] GT:{}, spatial detected:{} (p={}%): {}".format(content_type_mapping[category], content_type_mapping[index], prob, hit_or_miss)

if __name__ == "__main__":
    main()
