package com.htc.speedo.caffe;

/**
 * JUnit test for the jni solver wrapper in single precision.
 * @author Zhongyang Zheng (zhongyang_zheng@htc.com)
 */
public class JNIFloatTest extends JNITest {
  public JNIFloatTest() {
    super(1e-3, "%f", false);
  }
}
