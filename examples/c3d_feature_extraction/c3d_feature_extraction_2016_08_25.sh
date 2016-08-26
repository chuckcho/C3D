#!/usr/bin/env bash

VIDEODIR=/media/6TB2/Videos/test-streams

GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  ./prototxt/c3d_feature_extractor.prototxt \
  ./conv3d_deepnetA_sport1m_iter_1900000 \
  0 80 10000000 \
  ./prototxt/testvideos_output2.txt \
  fc6-1
