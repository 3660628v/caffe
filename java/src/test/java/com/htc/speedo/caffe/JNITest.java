package com.htc.speedo.caffe;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import caffe.Caffe.BlobProto;
import caffe.Caffe.DataParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import com.google.protobuf.Message.Builder;
import com.google.protobuf.TextFormat;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * JUnit test base class using the jni solver wrapper.
 * @author Zhongyang Zheng (zhongyang_zheng@htc.com)
 */
public abstract class JNITest {
  /** Load text format protobuf message. */
  protected void loadMessage(String path, Builder builder) throws IOException {
    InputStream istream = Files.newInputStream(Paths.get(path));
    InputStreamReader reader =
      new InputStreamReader(istream, StandardCharsets.UTF_8);
    TextFormat.merge(reader, builder);
    reader.close();
    istream.close();
  }

  /** Create a new solver for MNIST. */
  protected Solver loadMnistSolver() throws IOException {
    return loadMnistSolver(default_batch_size, 1);
  }

  /** Create a new solver for MNIST with custom batch size and iter size. */
  protected Solver loadMnistSolver(int batch_size, int iter_size) throws IOException {
    SolverParameter.Builder solver_builder = SolverParameter.newBuilder();
    loadMessage("examples/mnist/lenet_solver.prototxt", solver_builder);
    // skip test during train
    solver_builder.setTestInterval(Integer.MAX_VALUE).setTestInitialization(false);
    // set iter size
    solver_builder.setIterSize(iter_size);
    NetParameter.Builder net_builder = NetParameter.newBuilder();
    loadMessage("examples/mnist/lenet_train_test.prototxt", net_builder);
    if (batch_size != default_batch_size) {
      // override training batch size
      net_builder.getLayerBuilder(0).getDataParamBuilder().setBatchSize(batch_size);
    }
    Solver solver = new Solver();
    solver.init(solver_builder.build(), net_builder.build(), double_precision);
    return solver;
  }

  // accuracy for training loss
  private final double loss_accuracy;
  // accuracy for net parameter operation
  static private final float operation_accuracy = 0;
  // default batch size for MNIST
  static protected final int default_batch_size = 64;
  // The format used to output float numbers
  private final String double_format;
  // Create double precision c++ solver or not
  private final boolean double_precision;

  public JNITest(double loss_accuracy, String double_format, boolean double_precision) {
    this.loss_accuracy = loss_accuracy;
    this.double_format = double_format;
    this.double_precision = double_precision;
  }

  static {
    Solver.DisableGoogleLogging();
  }

  @Test
  public void testCPUGPUSolver() throws IOException {
    if (Solver.deviceCount > 0) { // have GPU
      // make sure CPU and GPU test starts from same weights
      Solver solver = loadMnistSolver();
      byte[] weight = solver.getWeight();
      solver.dispose();

      Solver.setDevice(-1);
      Assert.assertEquals(Solver.getDevice(), -1);
      System.out.println("Running test on CPU:");
      long cpuTime = testSolver(weight); // Test on CPU
      // switch to GPU
      Solver.setDevice(Solver.deviceCount - 1);
      Assert.assertEquals(Solver.getDevice(), Solver.deviceCount - 1);
      System.out.println("Running test on GPU:");
      long gpuTime = testSolver(weight); // Test on GPU
      // GPU should be faster than CPU
      Assert.assertTrue(gpuTime < cpuTime);
      System.out.printf("GPU time:\t%d us\tCPU time: %d us\n", gpuTime / 1000, cpuTime / 1000);
    }
    else { // CPU only
      Assert.assertEquals(Solver.getDevice(), -1);
      testSolver(null);
    }
  }

  private long testSolver(byte[] weight) throws IOException {
    long startTime = System.nanoTime();
    Solver solver = loadMnistSolver();
    if (weight != null) solver.setWeight(weight);
    // Initial net weight
    NetParameter weight1 = solver.getWeightProto();
    NetParameter delta1 = solver.getDeltaProto();
    // Train for 20 iterations
    double loss = solver.train(20);
    System.out.printf("Average loss of first 20 iterations is: " + double_format + "\n", loss);
    // Test for the whole test dataset
    double accuracy20 = solver.test(100);

    // Set weight back to the initial state
    solver.setWeight(weight1);
    // Test for the whole test dataset for initial state
    double accuracy0 = solver.test(100);
    Assert.assertTrue(accuracy20 > accuracy0);

    // set momentum to 0
    solver.updateParameter(SolverParameter.newBuilder().setMomentum(0).build());
    // Set weight to the same initial state
    solver.setWeight(weight1);
    System.out.printf("Loss of momentum = 0 is: " + double_format + "\n", solver.train(20));
    // test accuracy
    double accuracy1 = solver.test(100);
    Assert.assertTrue(accuracy1 < accuracy20);

    // reset solver parameter
    solver.updateParameter(SolverParameter.newBuilder().build());
    // set batch size to 1 (original is 64)
    solver.setBatchSize(1);
    // Set weight to the same initial state
    solver.setWeight(weight1);
    System.out.printf("Loss of batch size = 1 is: " + double_format + "\n", solver.train(20));
    // test accuracy
    accuracy1 = solver.test(100);
    Assert.assertTrue(accuracy1 < accuracy20);

    // Release the solver in c++
    solver.dispose();
    long endTime = System.nanoTime();
    return endTime - startTime;
  }

  @Test
  public void testGradientOnlySynchronous() throws IOException {
    // Use GPU if available
    Solver.setDevice(Solver.deviceCount - 1);
    int[] workers = {1, 2, 4, 8, 16, 32, 64};
    Solver solver = loadMnistSolver();
    // we use the same initial weight for all tests, so their final loss should be same
    byte[] weight = solver.getWeight();
    solver.dispose();
    int diff = 0;
    for (int worker: workers) {
      diff += singleWorkerGradientOnlySynchronous(worker, weight);
    }
    Assert.assertEquals(diff, 0);
  }

  int singleWorkerGradientOnlySynchronous(int numWorker, byte[] initWeight) throws IOException {
    // calculate batch size for each worker
    int batch_size = (default_batch_size - 1) / numWorker + 1;

    // This solver trains normally as comparison
    Solver solver = loadMnistSolver(batch_size, numWorker);
    solver.setWeight(initWeight);

    // master solver, used to merge deltas
    Solver master = loadMnistSolver(batch_size, 1);
    master.setWeight(initWeight);
    // worker solver, we use 1 solver, not many, to make sure it's completely same with single caffe
    Solver worker = loadMnistSolver(batch_size, 1);
    byte[] weight = initWeight, delta;
    double loss = -1, loss_normal = -1;
    int diff = 0;
    // train for 30 iterations
    for (int iter = 0; iter < 30; iter++) {
      delta = null;
      loss = 0;
      worker.setWeight(weight);
      for (int i = 0; i < numWorker; i++) {
        // worker.train calculates gradient, but no weight update. It clears diff at the beginning,
        // so invoking train on 1 worker n times, should be same like invoking on n workers.
        // Use 1 worker to make sure we visit excatly the same training examples as normal train.
        loss += worker.train(1, false);
        // sums all deltas
        if (delta == null)
          delta = worker.getDelta();
        else
          delta = NetParameterOperation.plus(delta, worker.getDelta());
      }
      loss /= numWorker;
      // update weight (and momentum) here
      weight = master.mergeDelta(NetParameterOperation.divide(delta, numWorker));
      loss_normal = solver.train(1);
      if (Math.abs(loss_normal - loss) >= loss_accuracy) {
        System.out.printf("%d workers (%d): Gradient only: " + double_format +
          "\tNormal: " + double_format + "\tDiff: " + double_format + "\n",
          numWorker, iter, loss, loss_normal, Math.abs(loss_normal - loss));
        diff++;
      }
    }
    System.out.printf("%d workers: Gradient only: " + double_format + "\tNormal: " +
      double_format + "\n", numWorker, loss, loss_normal);
    solver.dispose();
    master.dispose();
    worker.dispose();
    return diff;
  }

  @Test
  public void testNetParameterOpation() throws IOException {
    Solver solver = loadMnistSolver();
    // Initial net weight
    NetParameter weight1 = solver.getWeightProto();
    NetParameter delta1 = solver.getDeltaProto();
    // Train for 2 iterations
    double loss = solver.train(2);
    // Weight after training (different with inital weights)
    NetParameter weight2 = solver.getWeightProto();
    NetParameter delta2 = solver.getDeltaProto();

    // Test NetParameter operations
    NetParameter plus = NetParameterOperation.plus(weight1, weight2);
    for (int i = 0; i < weight1.getLayerCount(); i++) {
      LayerParameter layer1 = weight1.getLayer(i);
      LayerParameter layer20 = weight2.getLayer(i);
      LayerParameter layer = plus.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob20 = layer20.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        for (int k = 0; k < blob1.getDataCount(); k++) {
          Assert.assertEquals(blob1.getData(k) + blob20.getData(k), blob.getData(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDataCount(); k++) {
          Assert.assertEquals(blob1.getDoubleData(k) + blob20.getDoubleData(k), blob.getDoubleData(k), operation_accuracy);
        }
      }
    }
    NetParameter minus = NetParameterOperation.minus(weight1, weight2);
    for (int i = 0; i < weight1.getLayerCount(); i++) {
      LayerParameter layer1 = weight1.getLayer(i);
      LayerParameter layer20 = weight2.getLayer(i);
      LayerParameter layer = minus.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob20 = layer20.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        for (int k = 0; k < blob1.getDataCount(); k++) {
          Assert.assertEquals(blob1.getData(k) - blob20.getData(k), blob.getData(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDataCount(); k++) {
          Assert.assertEquals(blob1.getDoubleData(k) - blob20.getDoubleData(k), blob.getDoubleData(k), operation_accuracy);
        }
      }
    }
    NetParameter multiply = NetParameterOperation.multiply(weight1, 10.0);
    for (int i = 0; i < weight1.getLayerCount(); i++) {
      LayerParameter layer1 = weight1.getLayer(i);
      LayerParameter layer = multiply.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        Assert.assertTrue((blob1.getDataCount() > 0 && blob1.getDoubleDataCount() == 0) ||
          (blob1.getDataCount() == 0 && blob1.getDoubleDataCount() > 0));
        Assert.assertEquals(blob1.getDiffCount(), 0);
        Assert.assertEquals(blob1.getDoubleDiffCount(), 0);
        for (int k = 0; k < blob1.getDataCount(); k++) {
          Assert.assertEquals(blob1.getData(k) * 10, blob.getData(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDataCount(); k++) {
          Assert.assertEquals(blob1.getDoubleData(k) * 10, blob.getDoubleData(k), operation_accuracy);
        }
      }
    }
    NetParameter divide = NetParameterOperation.divide(weight1, 5.5);
    for (int i = 0; i < weight1.getLayerCount(); i++) {
      LayerParameter layer1 = weight1.getLayer(i);
      LayerParameter layer = divide.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        Assert.assertTrue((blob1.getDataCount() > 0 && blob1.getDoubleDataCount() == 0) ||
          (blob1.getDataCount() == 0 && blob1.getDoubleDataCount() > 0));
        Assert.assertEquals(blob1.getDiffCount(), 0);
        Assert.assertEquals(blob1.getDoubleDiffCount(), 0);
        for (int k = 0; k < blob1.getDataCount(); k++) {
          Assert.assertEquals(blob1.getData(k) / 5.5f, blob.getData(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDataCount(); k++) {
          Assert.assertEquals(blob1.getDoubleData(k) / 5.5, blob.getDoubleData(k), operation_accuracy);
        }
      }
    }
    // Test NetParameter operations for diffs
    plus = NetParameterOperation.plus(delta1, delta2);
    for (int i = 0; i < delta1.getLayerCount(); i++) {
      LayerParameter layer1 = delta1.getLayer(i);
      LayerParameter layer20 = delta2.getLayer(i);
      LayerParameter layer = plus.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob20 = layer20.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        for (int k = 0; k < blob1.getDiffCount(); k++) {
          Assert.assertEquals(blob1.getDiff(k) + blob20.getDiff(k), blob.getDiff(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDiffCount() ; k++) {
          Assert.assertEquals(blob1.getDoubleDiff(k) + blob20.getDoubleDiff(k), blob.getDoubleDiff(k), operation_accuracy);
        }
      }
    }
    minus = NetParameterOperation.minus(delta1, delta2);
    for (int i = 0; i < delta1.getLayerCount(); i++) {
      LayerParameter layer1 = delta1.getLayer(i);
      LayerParameter layer20 = delta2.getLayer(i);
      LayerParameter layer = minus.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob20 = layer20.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        for (int k = 0; k < blob1.getDiffCount(); k++) {
          Assert.assertEquals(blob1.getDiff(k) - blob20.getDiff(k), blob.getDiff(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDiffCount(); k++) {
          Assert.assertEquals(blob1.getDoubleDiff(k) - blob20.getDoubleDiff(k), blob.getDoubleDiff(k), operation_accuracy);
        }
      }
    }
    multiply = NetParameterOperation.multiply(delta1, 10.0);
    for (int i = 0; i < delta1.getLayerCount(); i++) {
      LayerParameter layer1 = delta1.getLayer(i);
      LayerParameter layer = multiply.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        Assert.assertTrue((blob1.getDiffCount() > 0 && blob1.getDoubleDiffCount() == 0) ||
          (blob1.getDiffCount() == 0 && blob1.getDoubleDiffCount() > 0));
        Assert.assertEquals(blob1.getDataCount(), 0);
        Assert.assertEquals(blob1.getDoubleDataCount(), 0);
        for (int k = 0; k < blob1.getDiffCount(); k++) {
          Assert.assertEquals(blob1.getDiff(k) * 10, blob.getDiff(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDiffCount(); k++) {
          Assert.assertEquals(blob1.getDoubleDiff(k) * 10, blob.getDoubleDiff(k), operation_accuracy);
        }
      }
    }
    divide = NetParameterOperation.divide(delta1, 5.5);
    for (int i = 0; i < delta1.getLayerCount(); i++) {
      LayerParameter layer1 = delta1.getLayer(i);
      LayerParameter layer = divide.getLayer(i);
      for (int j = 0; j < layer1.getBlobsCount(); j++) {
        BlobProto blob1 = layer1.getBlobs(j);
        BlobProto blob = layer.getBlobs(j);
        Assert.assertTrue((blob1.getDiffCount() > 0 && blob1.getDoubleDiffCount() == 0) ||
          (blob1.getDiffCount() == 0 && blob1.getDoubleDiffCount() > 0));
        Assert.assertEquals(blob1.getDataCount(), 0);
        Assert.assertEquals(blob1.getDoubleDataCount(), 0);
        for (int k = 0; k < blob1.getDiffCount(); k++) {
          Assert.assertEquals(blob1.getDiff(k) / 5.5f, blob.getDiff(k), operation_accuracy);
        }
        for (int k = 0; k < blob1.getDoubleDiffCount(); k++) {
          Assert.assertEquals(blob1.getDoubleDiff(k) / 5.5, blob.getDoubleDiff(k), operation_accuracy);
        }
      }
    }
  }
}
