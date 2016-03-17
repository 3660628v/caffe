package com.htc.speedo.caffe;

import caffe.Caffe.NetParameter;

import com.google.protobuf.InvalidProtocolBufferException;

public class NetParameterOperation {
  static {
    System.loadLibrary("caffe_jni");
  }

  private NetParameterOperation() {}

  /** Plus weight by weight for two net parameters. */
  static public NetParameter plus(NetParameter p1, NetParameter p2)
    throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(plus(p1.toByteArray(), p2.toByteArray()));
  }

  /**
   * Plus weight by weight for two net parameters with random skip layers.
   * @param skip The possibility to skip a layer. If a layer is skipped, value
   * of p1 is returned in the result for this layer.
   */
  static public NetParameter plus(NetParameter p1, NetParameter p2, float skip)
    throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(plus(p1.toByteArray(), p2.toByteArray(), skip));
  }

  /** Substract weight by weight for two net parameters. */
  static public NetParameter minus(NetParameter p1, NetParameter p2)
    throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(minus(p1.toByteArray(), p2.toByteArray()));
  }

  /** Multiply weights by a factor. */
  static public NetParameter multiply(NetParameter p, double factor)
    throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(multiply(p.toByteArray(), factor));
  }

  /** Divide weights by a factor. */
  static public NetParameter divide(NetParameter p, double factor)
    throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(divide(p.toByteArray(), factor));
  }

  // jni functions, see solver_jni.cpp for implementation

  /** Plus weight by weight for two net parameters. */
  static public native byte[] plus(byte[] model1, byte[] model2);
  /** Plus weight by weight for two net parameters with random skip layers. */
  static public native byte[] plus(byte[] model1, byte[] model2, float skip);
  /** Substract weight by weight for two net parameters. */
  static public native byte[] minus(byte[] model1, byte[] model2);
  /** Multiply weights by a factor. */
  static public native byte[] multiply(byte[] model, double factor);
  /** Divide weights by a factor. */
  static public native byte[] divide(byte[] model, double factor);
}
