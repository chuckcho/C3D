#!/usr/bin/env python

'''
Using fc activation features from C3D network (pretrained to UCF101 activity
recognition task)
'''

import numpy as np
import array
import sys
import os
import pylab
import glob
#from tsne import bh_sne
from sklearn.manifold import TSNE as tsne

def extract_and_average_features(feature_dir, feature='fc6-1'):
    # allow overwrite
    force_overwite = True
    #force_overwite = False
    clips_to_average = 4 # 2.135 second (fps=29.97)

    for dirname in os.listdir(feature_dir):
        clip_fullpath = os.path.join(feature_dir, dirname)
        if os.path.isdir(clip_fullpath):
            print "-"*79
            print "[Info] Processing shot={}...".format(dirname)

            clips = sorted(glob.glob(os.path.join(
                           clip_fullpath, '*.' + feature)),
                           key=lambda z:int(os.path.basename(z)[:-(len(feature)+1)]))

            num_clips = len(clips)
            for clip_count, clip in enumerate(clips):
                frame_num = int(os.path.basename(clip)[:-(len(feature)+1)])
                print "frame={}".format(frame_num)
                #clip = os.path.join(clip_fullpath, clip)
                f = open(clip, "rb") # read binary data
                s = f.read() # read all bytes into a string
                f.close()
                (n, c, l, h, w) = array.array("i", s[:20])
                feature_vec = np.array(array.array("f", s[20:]))

                #print "n={}, c={}, l={}, h={}, w={}".format(n, c, l, h, w)
                #print "feature_vec[:5]={}".format(feature_vec[:5])

                if clip_count % clips_to_average == 0:
                    start_frame_num = frame_num
                    avg_feature_vec = feature_vec
                else:
                    avg_feature_vec += feature_vec

                    # save
                    if (clip_count+1) % clips_to_average == 0:
                        avg_feature_vec /= clips_to_average
                        #print "avg_feature_vec[:5]={}, num_clips={}".format(avg_feature_vec[:5], num_clips)

                        shot_feature_filename = os.path.join(
                                clip_fullpath + '_f{0:05d}_to_f{1:05d}.csv'.format(
                                        start_frame_num,
                                        frame_num))
                        if os.path.isfile(shot_feature_filename) and not force_overwite:
                            print "[Warning] feature was already saved. Skipping this shot..."
                            continue
                        else:
                            print "[Info] saving {} feature as {}".format(feature, shot_feature_filename)
                            tmp = avg_feature_vec.reshape(1, avg_feature_vec.shape[0])
                            #np.savetxt(shot_feature_filename, avg_feature_vec);
                            np.savetxt(shot_feature_filename, tmp, fmt='%.16f', delimiter=',')

def extract_features(feature_dir, feature='fc6-1'):
    # allow overwrite
    force_overwite = True

    feature_file_pattern = os.path.join(feature_dir, '*.'+feature)
    for clip_count, clip in enumerate(glob.glob(feature_file_pattern)):
        print "-"*79
        print "[Info] Processing video={}...".format(clip)
        f = open(clip, "rb") # read binary data
        s = f.read() # read all bytes into a string
        f.close()
        feature_vec = np.matrix(array.array("f", s[20:]))
        #print "[debug] feature_vec={}".format(feature_vec)
        shot_feature_filename = clip.replace(feature, 'csv')
        if os.path.isfile(shot_feature_filename) and not force_overwite:
            print "[Warning] feature was already saved. Skipping this shot..."
            continue
        else:
            print "[Info] saving {} feature as {}".format(feature, shot_feature_filename)
            np.savetxt(shot_feature_filename, feature_vec, fmt='%.16f', delimiter=',')

def main():
    #feature_dir = '/media/6TB/Videos/youtube-dog-videos-for-demo-2016-Jun/C3D_features'
    feature_dir = '/media/6TB2/Videos/test-streams/C3D_features'

    #extract_and_average_features(feature_dir, feature='fc6-1')
    extract_features(feature_dir, feature='fc6-1')

if __name__ == "__main__":
    main()
