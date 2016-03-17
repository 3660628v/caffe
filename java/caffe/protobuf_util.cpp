#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "protobuf_util.hpp"

using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyInputStream;

bool ReadProtoFromByteArray(JNIEnv* env, jbyteArray array, Message* proto) {
  jsize size = env->GetArrayLength(array);
  jbyte* carray = reinterpret_cast<jbyte*>(
    env->GetPrimitiveArrayCritical(array, 0));

  // modified from ReadProtoFromBinaryFile function in caffe's io.cpp
  ZeroCopyInputStream* raw_input = new ArrayInputStream(carray, size);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  // the array is read-only, so jvm can safely discard the copy
  env->ReleasePrimitiveArrayCritical(array, carray, JNI_ABORT);
  delete coded_input;
  delete raw_input;
  return success;
}

jbyteArray WriteProtoToByteArray(JNIEnv* env, const Message& proto) {
  int size = proto.ByteSize();
  // create a byte[] object
  jbyteArray array = env->NewByteArray(size);
  jbyte* carray = reinterpret_cast<jbyte*>(
    env->GetPrimitiveArrayCritical(array, 0));
  proto.SerializeToArray(carray, size);
  // copy back the content and free the c++ array
  env->ReleasePrimitiveArrayCritical(array, carray, 0);
  return array;
}
