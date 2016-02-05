#!/bin/bash
# This script must be run with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"
# Install apt packages where the Ubuntu 12.04 default and ppa works for Caffe

# This ppa is for gflags and glog
add-apt-repository -y ppa:tuleu/precise-backports
apt-get -y update
apt-get install \
    wget git curl \
    python-dev python-numpy python3-dev\
    libleveldb-dev libsnappy-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev libgflags-dev libgoogle-glog-dev \
    bc

#    libopencv-dev \

# Install CUDA, if needed
if $WITH_CUDA; then
  CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_6.5-14_amd64.deb
  CUDA_FILE=/tmp/cuda_install.deb
  curl $CUDA_URL -o $CUDA_FILE
  dpkg -i $CUDA_FILE
  rm -f $CUDA_FILE
  apt-get -y update
  # Install the minimal CUDA subpackages required to test Caffe build.
  # For a full CUDA installation, add 'cuda' to the list of packages.
  apt-get -y install cuda-core-6-5 cuda-cublas-6-5 cuda-cublas-dev-6-5 cuda-cudart-6-5 cuda-cudart-dev-6-5 cuda-curand-6-5 cuda-curand-dev-6-5
  # Create CUDA symlink at /usr/local/cuda
  # (This would normally be created by the CUDA installer, but we create it
  # manually since we did a partial installation.)
  ln -s /usr/local/cuda-6.5 /usr/local/cuda
fi

# Install LMDB
LMDB_URL=https://github.com/LMDB/lmdb/archive/LMDB_0.9.14.tar.gz
LMDB_FILE=/tmp/lmdb.tar.gz
pushd .
wget $LMDB_URL -O $LMDB_FILE
tar -C /tmp -xzvf $LMDB_FILE
cd /tmp/lmdb*/libraries/liblmdb/
$MAKE
$MAKE install
popd
rm -f $LMDB_FILE

# Install the Python runtime dependencies via miniconda (this is much faster
# than using pip for everything).
export PATH=$CONDA_DIR/bin:$PATH
if [ ! -d $CONDA_DIR ]; then
  if [ "$PYTHON_VERSION" -eq "3" ]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  else
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  fi
  chmod +x miniconda.sh
  ./miniconda.sh -b -p $CONDA_DIR

  conda update --yes conda
  # The version of boost we're using for Python 3 depends on 3.4 for now.
  if [ "$PYTHON_VERSION" -eq "3" ]; then
    conda install --yes python=3.4
  fi
  conda install --yes numpy scipy matplotlib scikit-image pip
  # Let conda install boost (so that boost_python matches)
  conda install --yes -c https://conda.binstar.org/menpo boost=1.56.0
fi

# install protobuf 3 (just use the miniconda3 directory to avoid having to setup the path again)
if [ "$PYTHON_VERSION" -eq "3" ] && [ ! -e "$CONDA_DIR/bin/protoc" ]; then
  pushd .
  wget https://github.com/google/protobuf/archive/v3.0.0-alpha-3.1.tar.gz -O protobuf-3.tar.gz
  tar -C /tmp -xzvf protobuf-3.tar.gz
  cd /tmp/protobuf-3*/
  ./autogen.sh
  ./configure --prefix=$CONDA_DIR
  $MAKE
  $MAKE install
  popd
fi

if [ "$PYTHON_VERSION" -eq "3" ]; then
  pip install --pre protobuf
else
  pip install protobuf
fi

# install opencv3
apt-get -y update
apt-get install build-essential
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
cd /tmp/
wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/3.0.0/opencv-3.0.0.zip
unzip opencv-3.0.0.zip
cd opencv-3.0.0
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j
make install
