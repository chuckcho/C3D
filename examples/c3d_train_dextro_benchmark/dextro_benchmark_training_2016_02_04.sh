export GOOGLE_LOG_DIR=./log
export GLOG_DIR=./log
export GLOG_logtostderr=1

#../../build/tools/finetune_net.bin \
#  c3d_dextro_benchmark_solver_2016_02_04.prototxt \
#  ../../conv3d_deepnetA_sport1m_iter_1900000 \
#  2>&1 | tee c3d_dextro_benchmark_log_2016_02_04.txt

DATE=`date +%Y-%m-%d:%H:%M:%S`
echo ------------------------------------------ >> ./c3d_dextro_benchmark_log_2016_02_04.txt
echo New training has started at $DATE! >> ./c3d_dextro_benchmark_log_2016_02_04.txt
echo -e '\n' >> ./c3d_dextro_benchmark_log_2016_02_04.txt

../../build/tools/train_net.bin \
  ./c3d_dextro_benchmark_solver_2016_02_04.prototxt \
  ./c3d_dextro_benchmark_2016_02_04_iter_5000.solverstate \
  2>&1 \
  | tee -a ./c3d_dextro_benchmark_log_2016_02_04.txt
