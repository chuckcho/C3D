#!/usr/bin/env bash

VIDEODIR=/media/6TB/Videos/youtube-dog-videos-for-demo-2016-Jun
#mkdir -p ${VIDEODIR}/C3D_features

#GLOG_logtosterr=1 \
#  ../../build/tools/extract_image_features.bin \
#  prototxt/c3d_sport1m_feature_extractor_video.prototxt \
#  conv3d_deepnetA_sport1m_iter_1900000 \
#  0 50 1 \
#  prototxt/output_list_video_prefix.txt \
#  fc7-1 fc6-1 prob

GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  prototxt/c3d_youtube_demo_feature_extractor.prototxt \
  conv3d_deepnetA_sport1m_iter_1900000 \
  0 80 10000000 \
  prototxt/youtube_demo_output_list_video_prefix.txt \
  fc7-1 fc6-1 prob
