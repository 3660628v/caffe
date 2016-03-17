package com.htc.speedo.caffe;

import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import com.google.protobuf.InvalidProtocolBufferException;

/**
 * The solver jni wrapper for Caffe.
 * @author Zhongyang Zheng (zhongyang_zheng@htc.com)
 */
public class Solver {
  /** The total number of GPUs, 0 for CPU only mode. */
  public static final int deviceCount;

  static {
    System.loadLibrary("caffe_jni");
    deviceCount = getDeviceCount();
  }

  /** Disable google logging. All logs BEFORE this function are output to stderr. */
  public static native void DisableGoogleLogging();

  /** Set GPU device id in use, index start from 0. A negative device id indicates using CPU. */
  public static native void setDevice(int deviceId);

  /** Get GPU device id in use, index start from 0. A negative device id indicates using CPU. */
  public static native int getDevice();

  /** Creates a solver instance in java. Must init before use. */
  public Solver() {
    handle = 0;
  }

  /** Init the caffe solver with given parameters in c++. */
  public void init(SolverParameter solver, NetParameter model, boolean doublePrecision) {
    if (handle != 0)
      throw new IllegalStateException("Solver has already been initialized!");
    original_param = solver;
    handle = init(solver.toByteArray(), model.toByteArray(), doublePrecision);
  }

  /** Delete the caffe solver in c++. Should init again before train or test. */
  public void dispose() {
    dispose(handle);
    handle = 0;
  }

  /** Train for given iterations. Always update the deltas. */
  public double train(int iteration) {
    return train(iteration, true);
  }

  /**
   * Train for given iterations.
   * @param iteration Number of mini-batches
   * @param update_diff If set to false, the delta is not updated to the weights. You can get the
   * delta using getDelta function.
   */
  public double train(int iteration, boolean update_diff) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before train!");
    if (!update_diff && iteration > 1)
      throw new IllegalStateException("Cannot train more than one mini-batch if not update_diff!");
    return train(handle, iteration, update_diff);
  }

  /** Test for given iterations, should be (test data size/test batch size). */
  public double test(int iteration) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before test!");
    return test(handle, iteration);
  }

  /** Set iteration of the solver. */
  public void setIteration(int iteration) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before setIteration!");
    setIteration(handle, iteration);
  }

  /**
   * Set the batch size of input data layer to the given value. Only support DataLayer of TRAIN
   * phase, since we don't need to change batch size of the TEST phase input.
   * <P>
   * NOTE: If batch_size is 0, batch size is reset to the original one defined in model protobuf.
   */
  public void setBatchSize(int batch_size) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before setBatchSize!");
    if (batch_size < 0)
      throw new IllegalStateException("batch size must be non-negative in setBatchSize!");
    setBatchSize(handle, batch_size);
  }

  /**
   * Update solver paramter. Only parameters of momentum, weight decay and learning rates are
   * updated (see below list). The new_param does not need to provide every field, as parameters
   * passed in the {@link #init(caffe.Caffe.SolverParameter,caffe.Caffe.NetParameter,boolean) init function}
   * is used as default if the field in new_param is not modified. In this way, passing in an empty
   * {@link SolverParameter} will reset the solver parameters to the state after
   * {@link #init(caffe.Caffe.SolverParameter,caffe.Caffe.NetParameter,boolean) init function}.
   * <p>
   * Supported fields:
   * <ul>
   * <li>momentum</li>
   * <li>weight_decay</li>
   * <li>iter_size</li>
   * <li>base_lr</li>
   * <li>lr_policy</li>
   * <li>gamma</li>
   * <li>power</li>
   * <li>stepsize</li>
   * <li>stepvalue</li>
   * <li>delta</li>
   * <li>rms_decay</li>
   * </ul>
   */
  public void updateParameter(SolverParameter new_param) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before setBatchSize!");
    // use original_param as default if not provided in new_param
    SolverParameter.Builder builder = original_param.toBuilder().mergeFrom(new_param);
    // deal with repeated field stepvalue
    if (new_param.getStepvalueCount() > 0) {
      // if stepvalue is provided, we should only keep the ones in new_param
      builder.clearStepvalue();
      builder.addAllStepvalue(new_param.getStepvalueList());
    }
    updateParameter(handle, builder.build().toByteArray());
  }

  /** Set the caffe net parameters to the given model. */
  public void setWeight(NetParameter model) {
    setWeight(model.toByteArray());
  }

  /** Set the model. The model must be a valid caffe net parameter. */
  public void setWeight(byte[] model) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before setWeight!");
    setWeight(handle, model);
  }

  /** Get the caffe net parameters as protobuf. */
  public NetParameter getWeightProto() throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(getWeight());
  }

  /** Get the caffe net parameters. */
  public byte[] getWeight() {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before getWeight!");
    return getWeight(handle, false);
  }

  /** Set the caffe net parameter diffs to the given model and update to the model. */
  public NetParameter mergeDelta(NetParameter model) throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(mergeDelta(model.toByteArray()));
  }

  /** Set the caffe net parameter diffs to the given model and update to the model. */
  public byte[] mergeDelta(byte[] model) {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before mergeDelta!");
    return mergeDelta(handle, model);
  }

  /** Get the caffe net parameter gradients as protobuf. */
  public NetParameter getDeltaProto() throws InvalidProtocolBufferException {
    return NetParameter.parseFrom(getDelta());
  }

  /** Get the caffe net parameter gradients. */
  public byte[] getDelta() {
    if (handle == 0)
      throw new IllegalStateException("Solver must be initialized before getDelta!");
    return getWeight(handle, true);
  }

  // The c++ pointer for caffe solver
  private long handle;
  // The original solver parameter passed from Init
  private SolverParameter original_param;

  // jni functions, see solver_jni.cpp for implementation
  private static native int getDeviceCount();
  private static native long init(byte[] solver, byte[] model, boolean doublePrecision);
  private static native void dispose(long handle);
  private static native double train(long handle, int iteration, boolean update_diff);
  private static native double test(long handle, int iteration);
  private static native void setIteration(long handle, int iteration);
  private static native void setBatchSize(long handle, int batch_size);
  private static native void updateParameter(long handle, byte[] param);
  private static native void setWeight(long handle, byte[] weight);
  private static native byte[] getWeight(long handle, boolean diff);
  private static native byte[] mergeDelta(long handle, byte[] weight);
}
