GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  ./conv3d_ucf101_feature_extraction.prototxt \
  ./conv3d_ucf101_iter_60000 \
  0 \
  50 \
  837 \
  ./output.lst \
  fc6

# num of clips = 41822
# num of minibatches = 41822 / 50 = 836
