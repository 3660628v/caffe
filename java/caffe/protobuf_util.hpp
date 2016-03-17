#ifndef CAFFE_JNI_PROTOBUF_UTIL_HPP_
#define CAFFE_JNI_PROTOBUF_UTIL_HPP_

#include <google/protobuf/message.h>
#include <jni.h>

using google::protobuf::Message;

bool ReadProtoFromByteArray(JNIEnv* env, jbyteArray array, Message* proto);

jbyteArray WriteProtoToByteArray(JNIEnv* env, const Message& proto);

#endif
