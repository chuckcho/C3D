#!/usr/bin/env python

'''
Using fc6 activation features from C3D network (pretrained to UCF101 activity
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

c3d_feature_dir = '/media/6TB/Videos/youtube-dog-videos-for-demo-2016-Jun/C3D_features'

def extract_and_save_fc6_features(feature='fc6-1'):
    # allow overwrite
    force_overwite = True
    #force_overwite = False
    clips_to_average = 4 # 2.135 second (fps=29.97)

    for dirname in os.listdir(c3d_feature_dir):
        clip_fullpath = os.path.join(c3d_feature_dir, dirname)
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
                fc6 = np.array(array.array("f", s[20:]))

                #print "n={}, c={}, l={}, h={}, w={}".format(n, c, l, h, w)
                #print "fc6[:5]={}".format(fc6[:5])

                if clip_count % clips_to_average == 0:
                    start_frame_num = frame_num
                    fc6_avg = fc6
                else:
                    fc6_avg += fc6

                    # save
                    if (clip_count+1) % clips_to_average == 0:
                        fc6_avg /= clips_to_average
                        #print "fc6_avg[:5]={}, num_clips={}".format(fc6_avg[:5], num_clips)

                        shot_feature_filename = os.path.join(
                                clip_fullpath + '_f{0:05d}_to_f{1:05d}.csv'.format(
                                        start_frame_num,
                                        frame_num))
                        if os.path.isfile(shot_feature_filename) and not force_overwite:
                            print "[Warning] feature was already saved. Skipping this shot..."
                            continue
                        else:
                            print "[Info] saving fc6 feature as {}".format(shot_feature_filename)
                            tmp = fc6_avg.reshape(1, fc6_avg.shape[0])
                            #np.savetxt(shot_feature_filename, fc6_avg);
                            np.savetxt(shot_feature_filename, tmp, fmt='%.16f', delimiter=',')

def main():
    '''
    all_features_file = os.path.join(c3d_feature_dir, 'all_features.npy')
    all_labels_file   = os.path.join(c3d_feature_dir, 'all_labels.npy')

    if os.path.isfile(all_features_file) and os.path.isfile(all_labels_file):
        X = np.load(all_features_file)
        all_labels = np.load(all_labels_file)

    else:
        num_shots = 0
        all_labels = []
        for filename in os.listdir(c3d_feature_dir):

            if not filename.endswith('.csv'):
                continue

            c3d_feature_fullpath = os.path.join(c3d_feature_dir, filename)
            print "[Info] Processing shot={}...".format(c3d_feature_fullpath)
            c3d_feature = np.loadtxt(c3d_feature_fullpath)

            if num_shots == 0:
                X = c3d_feature
            else:
                X = np.vstack((X, c3d_feature))
            label = filename.split('_')[0]
            all_labels.append(label)
            num_shots += 1

            #print "X.shape={}, len(all_labels)={}, num_shots={}".format(X.shape, len(all_labels), num_shots)

        np.save(all_features_file, X)
        np.save(all_labels_file, all_labels)

    return X, all_labels
    '''
    extract_and_save_fc6_features(feature='fc6-1')

if __name__ == "__main__":
    main()
