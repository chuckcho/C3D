#!/usr/bin/env python

import cv2
import json
import sys

import os
import glob
import random

def get_num_frames(vid_file, image_extracted=False):

    if image_extracted:
        video_no_ext, ext = os.path.splitext(vid_file)
        image_files = glob.glob(os.path.join(video_no_ext, 'image_*.jpg'))
        return len(image_files)
    else:
        cap = cv2.VideoCapture(vid_file)
        if not cap.isOpened():
            print "[Warning] Can not open {}".format(vid_file)
            return -1
        num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
    return num_frames

def main():

    # video input
    video_base_dir = '/media/6TB/Videos/FCVID/FCVID_Video'

    # image_extracted?
    image_extracted = True

    # categories
    categories = [
            'outsideAirplane',
            'bird',
            'cat',
            'cow',
            'dog',
            'train'
            ]

    # overwrite train/val overwrite_train_val_files?
    overwrite_train_val_file = True

    # intermediate files
    train_file = './dextro_youtube_FCVID_train.txt'
    val_file = './dextro_youtube_FCVID_val.txt'

    # 4 means 4:1 = training samples:testing samples
    train_to_val_ratio = 4

    # min num of frames for segment
    c3d_depth = 16
    stride = c3d_depth * 3
    min_num_frames = 16

    if not os.path.isfile(train_file) or overwrite_train_val_file:
        save_train_val_files = True
    else:
        save_train_val_files = False

    if save_train_val_files:
        train_file_obj = open(train_file, "w")
        val_file_obj = open(val_file, "w")

    # valid frame range
    valid_frame_range = (0.3, 0.7)

    # iterate over all categories
    for category_id, category in enumerate(categories):
        print "[Info] Processing category: {}...".format(category)

        video_files = glob.glob(os.path.join(video_base_dir, category, '*.*'))

        # fix random seed
        random.seed('cerealkiller')
        random.shuffle(video_files)
        train_val_separating_index = int(float(len(video_files)) * train_to_val_ratio / (train_to_val_ratio+1))
        print "[Info] number of videos={}, train/val switching index={}".format(len(video_files), train_val_separating_index)

        for video_count, video_file in enumerate(video_files):
            num_frames = get_num_frames(video_file, image_extracted)
            print "[Info] Processing video: {}, num_frames={}...".format(video_file, num_frames)

            good_seg_start_frame = int(num_frames*valid_frame_range[0])
            good_seg_end_frame = int(num_frames*valid_frame_range[1])

            num_valid_frames = good_seg_end_frame-good_seg_start_frame+1

            #print "[Info] ({},{}), num_valid_frame={}...".format(good_seg_start_frame, good_seg_end_frame, num_valid_frames)

            if num_valid_frames < min_num_frames:
                print "[Info] Video too short. Skipping..."

            if image_extracted:
                video_file_or_dir, ext = os.path.splitext(video_file)
            else:
                video_file_or_dir = video_file

            for frame in range(good_seg_start_frame, good_seg_end_frame-c3d_depth+1, stride):
                text = "{} {} {}\n".format(video_file_or_dir, frame, category_id)

                # a very simple logic to split up train/val sets
                if save_train_val_files:
                    if video_count < train_val_separating_index:
                        train_file_obj.write(text)
                    else:
                        val_file_obj.write(text)


    print("-" * 79)

    if save_train_val_files:
        train_file_obj.close()
        val_file_obj.close()

if __name__ == "__main__":
    main()
