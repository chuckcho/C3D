GLOG_logtostderr=1 \
  ../../build/tools/train_net.bin \
  conv3d_ucf101_10_categories_solver.prototxt \
  2>&1 | tee c3d_ucf101_10_categories_train.log
