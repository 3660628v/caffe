#include "caffe/caffe.hpp"
#include "com_htc_speedo_caffe_NetParameterOperation.h"
#include "protobuf_util.hpp"

using caffe::BlobProto;
using caffe::LayerParameter;
using caffe::NetParameter;

jbyteArray Java_com_htc_speedo_caffe_NetParameterOperation_plus___3B_3B
  (JNIEnv* env, jclass obj, jbyteArray p1, jbyteArray p2) {
  NetParameter net1;
  ReadProtoFromByteArray(env, p1, &net1);
  NetParameter net2;
  ReadProtoFromByteArray(env, p2, &net2);

  // plus weights of p2 to p1
  int layerCount = net1.layer_size();
  for (int i = 0; i < layerCount; i++) {
    LayerParameter* layer1 = net1.mutable_layer(i);
    const LayerParameter& layer2 = net2.layer(i);
    int blobCount = layer1->blobs_size();
    for (int j = 0; j < blobCount; j++) {
      BlobProto* blob1 = layer1->mutable_blobs(j);
      const BlobProto& blob2 = layer2.blobs(j);
      // plus data field
      float* data = blob1->mutable_data()->mutable_data();
      int dataCount = blob2.data_size();
      for (int k = 0; k < dataCount; k++) {
        data[k] += blob2.data(k);
      }
      double* double_data = blob1->mutable_double_data()->mutable_data();
      dataCount = blob2.double_data_size();
      for (int k = 0; k < dataCount; k++) {
        double_data[k] += blob2.double_data(k);
      }
      // plus diff field
      float* diff = blob1->mutable_diff()->mutable_data();
      int diffCount = blob2.diff_size();
      for (int k = 0; k < diffCount; k++) {
        diff[k] += blob2.diff(k);
      }
      double* double_diff = blob1->mutable_double_diff()->mutable_data();
      diffCount = blob2.double_diff_size();
      for (int k = 0; k < diffCount; k++) {
        double_diff[k] += blob2.double_diff(k);
      }
    }
  }

  return WriteProtoToByteArray(env, net1);
}

jbyteArray Java_com_htc_speedo_caffe_NetParameterOperation_plus___3B_3BF
  (JNIEnv* env, jclass obj, jbyteArray p1, jbyteArray p2, jfloat skip) {
  NetParameter net1;
  ReadProtoFromByteArray(env, p1, &net1);
  NetParameter net2;
  ReadProtoFromByteArray(env, p2, &net2);

  // plus weights of p2 to p1
  int layerCount = net1.layer_size();
  float random_number;
  for (int i = 0; i < layerCount; i++) {
    LayerParameter* layer1 = net1.mutable_layer(i);
    const LayerParameter& layer2 = net2.layer(i);
    int blobCount = layer1->blobs_size();
    for (int j = 0; j < blobCount; j++) {
      // random skip layer
      caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &random_number);
      if (skip > random_number) continue;
      BlobProto* blob1 = layer1->mutable_blobs(j);
      const BlobProto& blob2 = layer2.blobs(j);
      // plus data field
      float* data = blob1->mutable_data()->mutable_data();
      int dataCount = blob2.data_size();
      for (int k = 0; k < dataCount; k++) {
        data[k] += blob2.data(k);
      }
      double* double_data = blob1->mutable_double_data()->mutable_data();
      dataCount = blob2.double_data_size();
      for (int k = 0; k < dataCount; k++) {
        double_data[k] += blob2.double_data(k);
      }
      // plus diff field
      float* diff = blob1->mutable_diff()->mutable_data();
      int diffCount = blob2.diff_size();
      for (int k = 0; k < diffCount; k++) {
        diff[k] += blob2.diff(k);
      }
      double* double_diff = blob1->mutable_double_diff()->mutable_data();
      diffCount = blob2.double_diff_size();
      for (int k = 0; k < diffCount; k++) {
        double_diff[k] += blob2.double_diff(k);
      }
    }
  }

  return WriteProtoToByteArray(env, net1);
}

jbyteArray Java_com_htc_speedo_caffe_NetParameterOperation_minus
  (JNIEnv* env, jclass obj, jbyteArray p1, jbyteArray p2) {
  NetParameter net1;
  ReadProtoFromByteArray(env, p1, &net1);
  NetParameter net2;
  ReadProtoFromByteArray(env, p2, &net2);

  // substract weights of p1 in-place by p2
  int layerCount = net1.layer_size();
  for (int i = 0; i < layerCount; i++) {
    LayerParameter* layer1 = net1.mutable_layer(i);
    const LayerParameter& layer2 = net2.layer(i);
    int blobCount = layer1->blobs_size();
    for (int j = 0; j < blobCount; j++) {
      BlobProto* blob1 = layer1->mutable_blobs(j);
      const BlobProto& blob2 = layer2.blobs(j);
      // minus data field
      float* data = blob1->mutable_data()->mutable_data();
      int dataCount = blob2.data_size();
      for (int k = 0; k < dataCount; k++) {
        data[k] -= blob2.data(k);
      }
      double* double_data = blob1->mutable_double_data()->mutable_data();
      dataCount = blob2.double_data_size();
      for (int k = 0; k < dataCount; k++) {
        double_data[k] -= blob2.double_data(k);
      }
      // minus diff field
      float* diff = blob1->mutable_diff()->mutable_data();
      int diffCount = blob2.diff_size();
      for (int k = 0; k < diffCount; k++) {
        diff[k] -= blob2.diff(k);
      }
      double* double_diff = blob1->mutable_double_diff()->mutable_data();
      diffCount = blob2.double_diff_size();
      for (int k = 0; k < diffCount; k++) {
        double_diff[k] -= blob2.double_diff(k);
      }
    }
  }

  return WriteProtoToByteArray(env, net1);
}

jbyteArray Java_com_htc_speedo_caffe_NetParameterOperation_multiply
  (JNIEnv* env, jclass obj, jbyteArray p, jdouble factor) {
  NetParameter net;
  ReadProtoFromByteArray(env, p, &net);

  // multiply weights of p in-place by factor
  int layerCount = net.layer_size();
  for (int i = 0; i < layerCount; i++) {
    LayerParameter* layer = net.mutable_layer(i);
    int blobCount = layer->blobs_size();
    for (int j = 0; j < blobCount; j++) {
      // multiply data field
      BlobProto* blob = layer->mutable_blobs(j);
      float* data = blob->mutable_data()->mutable_data();
      int dataCount = blob->data_size();
      for (int k = 0; k < dataCount; k++) {
        data[k] *= factor;
      }
      double* double_data = blob->mutable_double_data()->mutable_data();
      dataCount = blob->double_data_size();
      for (int k = 0; k < dataCount; k++) {
        double_data[k] *= factor;
      }
      // multiply diff field
      float* diff = blob->mutable_diff()->mutable_data();
      int diffCount = blob->diff_size();
      for (int k = 0; k < diffCount; k++) {
        diff[k] *= factor;
      }
      double* double_diff = blob->mutable_double_diff()->mutable_data();
      diffCount = blob->double_diff_size();
      for (int k = 0; k < diffCount; k++) {
        double_diff[k] *= factor;
      }
    }
  }

  return WriteProtoToByteArray(env, net);
}

jbyteArray Java_com_htc_speedo_caffe_NetParameterOperation_divide
  (JNIEnv* env, jclass obj, jbyteArray p, jdouble factor) {
  NetParameter net;
  ReadProtoFromByteArray(env, p, &net);

  // divide weights of p in-place by factor
  int layerCount = net.layer_size();
  for (int i = 0; i < layerCount; i++) {
    LayerParameter* layer = net.mutable_layer(i);
    int blobCount = layer->blobs_size();
    for (int j = 0; j < blobCount; j++) {
      // divide data field
      BlobProto* blob = layer->mutable_blobs(j);
      float* data = blob->mutable_data()->mutable_data();
      int dataCount = blob->data_size();
      for (int k = 0; k < dataCount; k++) {
        data[k] /= factor;
      }
      double* double_data = blob->mutable_double_data()->mutable_data();
      dataCount = blob->double_data_size();
      for (int k = 0; k < dataCount; k++) {
        double_data[k] /= factor;
      }
      // divide diff field
      float* diff = blob->mutable_diff()->mutable_data();
      int diffCount = blob->diff_size();
      for (int k = 0; k < diffCount; k++) {
        diff[k] /= factor;
      }
      double* double_diff = blob->mutable_double_diff()->mutable_data();
      diffCount = blob->double_diff_size();
      for (int k = 0; k < diffCount; k++) {
        double_diff[k] /= factor;
      }
    }
  }

  return WriteProtoToByteArray(env, net);
}
