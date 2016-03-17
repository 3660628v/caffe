package com.htc.speedo.caffe;

/**
 * JUnit test for the jni solver wrapper in double precision.
 * @author Zhongyang Zheng (zhongyang_zheng@htc.com)
 */
public class JNIDoubleTest extends JNITest {
  public JNIDoubleTest() {
    super(1e-14, "%.14f", true);
  }
}
