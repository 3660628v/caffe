#include <map>
#include <utility>

#include "caffe/caffe.hpp"
#include "com_htc_speedo_caffe_Solver.h"
#include "jni_solver.hpp"
#include "protobuf_util.hpp"

using caffe::Caffe;
using caffe::map;
using caffe::NetParameter;
using caffe::SolverParameter;

// A map to record if a solver is double precision or not
map<jlong, bool> precisionMap;

void Java_com_htc_speedo_caffe_Solver_DisableGoogleLogging
  (JNIEnv * env, jclass obj) {
  FLAGS_minloglevel = 4;  // FATAL is 3, set level to 4 ignores all logs
}

void Java_com_htc_speedo_caffe_Solver_setDevice(JNIEnv * env, jclass obj, jint device_id) {
  if (device_id < 0) {
    // only set mode for cpu, since SetDevice would fail in cpu only compilation
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(device_id);
  }
}

jint Java_com_htc_speedo_caffe_Solver_getDevice(JNIEnv * env, jclass obj) {
  if (Caffe::mode() == Caffe::GPU) {
#ifdef CPU_ONLY
    NO_GPU;  // if we are in cpu only compilation, the program would fail
#else
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));  // get device id from cuda
    return device_id;
#endif
  } else {
    return -1;  // return negative device id for CPU
  }
}

jint Java_com_htc_speedo_caffe_Solver_getDeviceCount(JNIEnv * env, jclass obj) {
#ifdef CPU_ONLY
    return 0;  // if we are in cpu only compilation, then no GPU available
#else
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));  // get device count from cuda
    return count;
#endif
}

jlong Java_com_htc_speedo_caffe_Solver_init(JNIEnv* env, jclass obj,
  jbyteArray solverBytes, jbyteArray modelBytes, jboolean doublePrecision) {
  // parse parameters
  SolverParameter solver_param;
  ReadProtoFromByteArray(env, solverBytes, &solver_param);
  NetParameter net_param;
  ReadProtoFromByteArray(env, modelBytes, &net_param);
  solver_param.mutable_net_param()->CopyFrom(net_param);
  solver_param.clear_net();

  jlong handle = 0;
  // create solver
  if (doublePrecision) {
    handle = (jlong) new JNISolver<double>(solver_param);
  } else {
    handle = (jlong) new JNISolver<float>(solver_param);
  }
  precisionMap.insert(std::pair<jlong, bool>(handle, doublePrecision));
  return handle;
}

void Java_com_htc_speedo_caffe_Solver_dispose(JNIEnv* env, jclass obj, jlong handle) {
  if (precisionMap[handle]) {
    delete (JNISolver<double>*) handle;
  } else {
    delete (JNISolver<float>*) handle;
  }
}

jdouble Java_com_htc_speedo_caffe_Solver_train(JNIEnv* env, jclass obj,
  jlong handle, jint iteration, jboolean update_diff) {
  // train for given iterations
  if (precisionMap[handle]) {
    return ((JNISolver<double>*) handle)->train(iteration, update_diff);
  } else {
    return ((JNISolver<float>*) handle)->train(iteration, update_diff);
  }
}

jdouble Java_com_htc_speedo_caffe_Solver_test(JNIEnv* env, jclass obj,
  jlong handle, jint iteration) {
  // test for given iterations, should be (test data size/test batch size).
  if (precisionMap[handle]) {
    return ((JNISolver<double>*) handle)->test(iteration);
  } else {
    return ((JNISolver<float>*) handle)->test(iteration);
  }
}

void Java_com_htc_speedo_caffe_Solver_setIteration(JNIEnv* env, jclass obj,
  jlong handle, jint iteration) {
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->setIteration(iteration);
  } else {
    ((JNISolver<float>*) handle)->setIteration(iteration);
  }
}

void Java_com_htc_speedo_caffe_Solver_setBatchSize(JNIEnv* env, jclass obj,
  jlong handle, jint batch_size) {
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->setBatchSize(batch_size);
  } else {
    ((JNISolver<float>*) handle)->setBatchSize(batch_size);
  }
}

void Java_com_htc_speedo_caffe_Solver_updateParameter(JNIEnv* env, jclass obj,
  jlong handle, jbyteArray paramBytes) {
  SolverParameter param;
  ReadProtoFromByteArray(env, paramBytes, &param);
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->updateParameter(param);
  } else {
    ((JNISolver<float>*) handle)->updateParameter(param);
  }
}

void Java_com_htc_speedo_caffe_Solver_setWeight(JNIEnv* env, jclass obj,
  jlong handle, jbyteArray weight) {
  NetParameter net_param;
  ReadProtoFromByteArray(env, weight, &net_param);
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->setWeight(net_param);
  } else {
    ((JNISolver<float>*) handle)->setWeight(net_param);
  }
}

jbyteArray Java_com_htc_speedo_caffe_Solver_getWeight(JNIEnv* env, jclass obj,
  jlong handle, jboolean diff) {
  NetParameter net_param;
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->getWeight(&net_param, diff);
  } else {
    ((JNISolver<float>*) handle)->getWeight(&net_param, diff);
  }
  return WriteProtoToByteArray(env, net_param);
}

jbyteArray Java_com_htc_speedo_caffe_Solver_mergeDelta(JNIEnv* env, jclass obj,
  jlong handle, jbyteArray delta) {
  NetParameter net_param;
  ReadProtoFromByteArray(env, delta, &net_param);
  if (precisionMap[handle]) {
    ((JNISolver<double>*) handle)->mergeDelta(net_param, &net_param);
  } else {
    ((JNISolver<float>*) handle)->mergeDelta(net_param, &net_param);
  }
  return WriteProtoToByteArray(env, net_param);
}
