#!/bin/bash -e
# change variables here
working_dir=target
parcel_name=CAFFE
parcel_version=0.1

# Install on machines in the cluster:
# sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

# Install dependencies and build tools
sudo apt-get install -y gfortran cmake maven

# Script actions start here

# Get the script directory
script_dir=$(dirname $(readlink -f $0))
cd $script_dir

# the parcel top folder (the folder will be within working_dir)
parcel_dir=${parcel_name}-${parcel_version}

# create the parcel directory
rm -rf ${working_dir}/${parcel_dir}
mkdir -p ${working_dir}/${parcel_dir}
cd $working_dir

# Download and build dependencies
# Protobuf
if [ ! -d protobuf-2.5.0 ]; then
  wget https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
  tar -vxf protobuf-2.5.0.tar.gz
  rm protobuf-2.5.0.tar.gz
  cd protobuf-2.5.0
  ./configure
  make
else
  cd protobuf-2.5.0
fi
make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd ..

# OpenBLAS
if [ ! -d OpenBLAS-master ]; then
  wget https://github.com/xianyi/OpenBLAS/archive/master.zip -O OpenBLAS.zip
  unzip OpenBLAS.zip
  rm OpenBLAS.zip
  cd OpenBLAS-master
  make
else
  cd OpenBLAS-master
fi
make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd ..
mv ${parcel_dir}/opt/OpenBLAS/include/* ${parcel_dir}/usr/local/include
mv ${parcel_dir}/opt/OpenBLAS/lib/* ${parcel_dir}/usr/local/lib
rm -r ${parcel_dir}/opt

# glog
if [ ! -d glog-0.3.3 ]; then
  wget https://github.com/google/glog/archive/v0.3.3.tar.gz
  tar zxvf v0.3.3.tar.gz
  rm v0.3.3.tar.gz
  cd glog-0.3.3
  ./configure
  # Disable gflags during compilation
  sed -i "s/#define HAVE_LIB_GFLAGS 1//g" src/config.h
  sed -i "s_#include <gflags/gflags.h>__g" src/glog/logging.h
  make
else
  cd glog-0.3.3
fi
make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd ..

# gflags
if [ ! -d gflags-2.1.2 ]; then
  wget https://github.com/gflags/gflags/archive/v2.1.2.zip
  unzip v2.1.2.zip
  rm v2.1.2.zip
  cd gflags-2.1.2
  mkdir build && cd build
  CXXFLAGS="-fPIC" cmake ..
  CXXFLAGS="-fPIC" make
else
  cd gflags-2.1.2/build
fi
CXXFLAGS="-fPIC" make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd ../..

# lmdb
if [ ! -d lmdb-0.9.15 ]; then
  wget https://github.com/clibs/lmdb/archive/0.9.15.tar.gz
  tar zxvf 0.9.15.tar.gz
  rm 0.9.15.tar.gz
  cd lmdb-0.9.15
  make
else
  cd lmdb-0.9.15
fi
mkdir -p $script_dir/${working_dir}/${parcel_dir}/usr/local/man/man1
make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd ..

# Caffe
cd ../../..
make all java
make install DESTDIR=$script_dir/${working_dir}/${parcel_dir}
cd $script_dir/$working_dir
mv ${parcel_dir}/usr/lib/* ${parcel_dir}/usr/local/lib
rm -r ${parcel_dir}/usr/lib/

# Only keep the library files
cd ${parcel_dir}
mv usr/local/lib .
rm -r usr

# copy and modify meta data
cd $script_dir
cp -r meta ${working_dir}/${parcel_dir}/
cd ${working_dir}
sed -i "s/parcel_name/${parcel_name}/g;s/parcel_version/${parcel_version}/g" ${parcel_dir}/meta/parcel.json

# remove the old output (we only keep the latest parcel)
rm -rf output
mkdir output
codename=$(lsb_release -cs)
# create parcel file in output folder
tar zcvhf output/${parcel_dir}-$codename.parcel ${parcel_dir}/ --owner=root --group=root

# validate parcel and generate manifest infos
cd output
${script_dir}/../validator -f ${parcel_dir}-$codename.parcel
${script_dir}/../make_manifest .

# remove the parcel folder
rm -rf ../${parcel_dir}
