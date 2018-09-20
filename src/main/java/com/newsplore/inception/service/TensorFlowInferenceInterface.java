/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific top governing permissions and
limitations under the License.
==============================================================================*/

package com.newsplore.inception.service;


import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Tensors;
import org.tensorflow.types.UInt8;

import com.top.tf_obj_detect.TensorFlowObjectDetectionAPIModel;
import com.newsplore.util.StringUtil;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wrapper over the TensorFlow API ({@link Graph}, {@link Session}) providing a smaller API surface
 * for inference.
 *
 * <p>See tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java for an
 * example usage.
 */

public class TensorFlowInferenceInterface {
  private static final String TAG = "TensorFlowInferenceInterface";
  private StringUtil StringUtil = new StringUtil();
  private static final Logger log = LoggerFactory.getLogger(TensorFlowInferenceInterface.class);

  /*
   * Load a TensorFlow model from the AssetManager or from disk if it is not an asset file.
   *
   * @param assetManager The AssetManager to use to load the model file.
   * @param model The filepath to the GraphDef proto representing the model.
   */
  public TensorFlowInferenceInterface(String model) {
    prepareNativeRuntime();

    this.modelName = model;
    this.g = new Graph();
    this.sess = new Session(g);
    this.runner = sess.runner();

    InputStream is = null;
      // Perhaps the model file is not an asset but is on disk.
      try {
        is = new FileInputStream(model);
     
  
      byte[] graphDef = new byte[is.available()];
      final int numBytesRead = is.read(graphDef);
      if (numBytesRead != graphDef.length) {
    	  is.close();
        throw new IOException(
            "read error: read only "
                + numBytesRead
                + " of the graph, expected to read "
                + graphDef.length);
      }
      loadGraph(graphDef, g);
      is.close();
      log.info(TAG, "Successfully loaded model from '" + model + "'");
      } catch (IOException e2) {
          throw new RuntimeException("Failed to load model from '" + model + "'", e2);
        }
  }

  /*
   * Load a TensorFlow model from provided InputStream.
   * Note: The InputStream will not be closed after loading model, users need to
   * close it themselves.
   *
   * @param is The InputStream to use to load the model.
   */
  public TensorFlowInferenceInterface(InputStream is) {
    prepareNativeRuntime();

    // modelName is redundant for model loading from input stream, here is for
    // avoiding error in initialization as modelName is marked final.
    this.modelName = "";
    this.g = new Graph();
    this.sess = new Session(g);
    this.runner = sess.runner();

    try
    {    
      log.info("initializeTensorFlow");
      log.info("readGraphDef");
      int baosInitSize = is.available() > 16384 ? is.available() : 16384;
      ByteArrayOutputStream baos = new ByteArrayOutputStream(baosInitSize);
      int numBytesRead;
      byte[] buf = new byte[16384];
      while ((numBytesRead = is.read(buf, 0, buf.length)) != -1) {
        baos.write(buf, 0, numBytesRead);
      }
      byte[] graphDef = baos.toByteArray();

      loadGraph(graphDef, g);
      log.info("Successfully loaded model from the input stream");
      } catch (IOException e) {
      throw new RuntimeException("Failed to load model from the input stream", e);
    }
  }

  /*
   * Construct a TensorFlowInferenceInterface with provided Graph
   *
   * @param g The Graph to use to construct this interface.
   */
  public TensorFlowInferenceInterface(Graph g) {
    prepareNativeRuntime();

    // modelName is redundant here, here is for
    // avoiding error in initialization as modelName is marked final.
    this.modelName = "";
    this.g = g;
    this.sess = new Session(g);
    this.runner = sess.runner();
  }

  /**
   * Runs inference between the previously registered input nodes (via feed*) and the requested
   * output nodes. Output nodes can then be queried with the fetch* methods.
   *
   * @param outputNames A list of output nodes which should be filled by the inference pass.
   */
  public void run(String[] outputNames) {
    run(outputNames, false);
  }

  /**
   * Runs inference between the previously registered input nodes (via feed*) and the requested
   * output nodes. Output nodes can then be queried with the fetch* methods.
   *
   * @param outputNames A list of output nodes which should be filled by the inference pass.
   */
  public void run(String[] outputNames, boolean enableStats) {
    run(outputNames, enableStats, new String[] {});
  }

  /**
   * An overloaded version of runInference that allows supplying targetNodeNames as well
   */
  public void run(String[] outputNames, boolean enableStats, String[] targetNodeNames) {
    // Release any Tensors from the previous run calls.
    closeFetches();

    // Add fetches.
    for (String o : outputNames) {
      fetchNames.add(o);
      TensorId tid = TensorId.parse(o);
      runner.fetch(tid.name, tid.outputIndex);
    }

	// Add targets.
    for (String t : targetNodeNames) {
      runner.addTarget(t);
    }

    // Run the session.
    try {
      if (enableStats) {
    	  /**
    	   * source code is :  runner.setOptions(RunStats.runOptions())
    	   * 
    	   *  The code copied from 'https://github.com/baidu-research/tensorflow-allreduce/blob/master/tensorflow/contrib/android/java/org/tensorflow/contrib/android/RunStats.java' 
    	   */
    	
        Session.Run r = runner.setOptions(runStats).runAndFetchMetadata();
        fetchTensors = r.outputs;
        byte[] c = new byte[runStats.length + r.metadata.length];
        System.arraycopy(runStats, 0, c, 0, runStats.length);
        System.arraycopy( r.metadata, 0, c, runStats.length, r.metadata.length);
        runStats = c.clone();
      } else {
        fetchTensors = runner.run();
      }
    } catch (RuntimeException e) {
      // Ideally the exception would have been let through, but since this interface predates the
      // TensorFlow Java API, must return -1.
    	
     log.info(
          "Failed to run TensorFlow inference with inputs:["
              + StringUtil.join(", ", feedNames)
              + "], outputs:["
              + StringUtil.join(", ", fetchNames)
              + "]");
      throw e;
    } finally {
      // Always release the feeds (to save resources) and reset the runner, this run is
      // over.
      closeFeeds();
      runner = sess.runner();
    }
  }

  /** Returns a reference to the Graph describing the computation run during inference. */
  public Graph graph() {
    return g;
  }

  public Operation graphOperation(String operationName) {
    final Operation operation = g.operation(operationName);
    if (operation == null) {
      throw new RuntimeException(
          "Node '" + operationName + "' does not exist in model '" + modelName + "'");
    }
    return operation;
  }

  /** Returns the last stat summary string if logging is enabled. */
  public String getStatString() {
    return (runStats == null) ? "" : runStats.toString();
  }

  /**
   * Cleans up the state associated with this Object.
   *
   * <p>The TenosrFlowInferenceInterface object is no longer usable after this method returns.
   */
  public void close() {
    closeFeeds();
    closeFetches();
    sess.close();
    g.close();
    runStats = null;
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  // Methods for taking a native Tensor and filling it with values from Java arrays.

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, boolean[] src, long... dims) {
    byte[] b = new byte[src.length];

    for (int i = 0; i < src.length; i++) {
      b[i] = src[i] ? (byte) 1 : (byte) 0;
    }

    addFeed(inputName, Tensor.create(Boolean.class, dims, ByteBuffer.wrap(b)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, float[] src, long... dims) {
    addFeed(inputName, Tensor.create(dims, FloatBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, int[] src, long... dims) {
    addFeed(inputName, Tensor.create(dims, IntBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, long[] src, long... dims) {
    addFeed(inputName, Tensor.create(dims, LongBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, double[] src, long... dims) {
    addFeed(inputName, Tensor.create(dims, DoubleBuffer.wrap(src)));
  }

  /**
   * Given a source array with shape {@link dims} and content {@link src}, copy the contents into
   * the input Tensor with name {@link inputName}. The source array {@link src} must have at least
   * as many elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, byte[] src, long... dims) {
    addFeed(inputName, Tensor.create(UInt8.class, dims, ByteBuffer.wrap(src)));
  }

  /**
   * Copy a byte sequence into the input Tensor with name {@link inputName} as a string-valued
   * scalar tensor. In the TensorFlow type system, a "string" is an arbitrary sequence of bytes, not
   * a Java {@code String} (which is a sequence of characters).
   */
  public void feedString(String inputName, byte[] src) {
    addFeed(inputName, Tensors.create(src));
  }

  /**
   * Copy an array of byte sequences into the input Tensor with name {@link inputName} as a
   * string-valued one-dimensional tensor (vector). In the TensorFlow type system, a "string" is an
   * arbitrary sequence of bytes, not a Java {@code String} (which is a sequence of characters).
   */
  public void feedString(String inputName, byte[][] src) {
    addFeed(inputName, Tensors.create(src));
  }

  // Methods for taking a native Tensor and filling it with src from Java native IO buffers.

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, FloatBuffer src, long... dims) {
    addFeed(inputName, Tensor.create(dims, src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, IntBuffer src, long... dims) {
    addFeed(inputName, Tensor.create(dims, src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, LongBuffer src, long... dims) {
    addFeed(inputName, Tensor.create(dims, src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, DoubleBuffer src, long... dims) {
    addFeed(inputName, Tensor.create(dims, src));
  }

  /**
   * Given a source buffer with shape {@link dims} and content {@link src}, both stored as
   * <b>direct</b> and <b>native ordered</b> java.nio buffers, copy the contents into the input
   * Tensor with name {@link inputName}. The source buffer {@link src} must have at least as many
   * elements as that of the destination Tensor. If {@link src} has more elements than the
   * destination has capacity, the copy is truncated.
   */
  public void feed(String inputName, ByteBuffer src, long... dims) {
    addFeed(inputName, Tensor.create(UInt8.class, dims, src));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   */
  public void fetch(String outputName, float[] dst) {
    fetch(outputName, FloatBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   */
  public void fetch(String outputName, int[] dst) {
    fetch(outputName, IntBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   */
  public void fetch(String outputName, long[] dst) {
    fetch(outputName, LongBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   */
  public void fetch(String outputName, double[] dst) {
    fetch(outputName, DoubleBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into a Java array. {@link
   * dst} must have length greater than or equal to that of the source Tensor. This operation will
   * not affect dst's content past the source Tensor's size.
   */
  public void fetch(String outputName, byte[] dst) {
    fetch(outputName, ByteBuffer.wrap(dst));
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   */
  public void fetch(String outputName, FloatBuffer dst) {
    getTensor(outputName).writeTo(dst);
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   */
  public void fetch(String outputName, IntBuffer dst) {
    getTensor(outputName).writeTo(dst);
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   */
  public void fetch(String outputName, LongBuffer dst) {
    getTensor(outputName).writeTo(dst);
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   */
  public void fetch(String outputName, DoubleBuffer dst) {
    getTensor(outputName).writeTo(dst);
  }

  /**
   * Read from a Tensor named {@link outputName} and copy the contents into the <b>direct</b> and
   * <b>native ordered</b> java.nio buffer {@link dst}. {@link dst} must have capacity greater than
   * or equal to that of the source Tensor. This operation will not affect dst's content past the
   * source Tensor's size.
   */
  public void fetch(String outputName, ByteBuffer dst) {
    getTensor(outputName).writeTo(dst);
  }

  private void prepareNativeRuntime() {
    log.info(TAG, "Checking to see if TensorFlow native methods are already loaded");
    try {
      // Hack to see if the native libraries have been loaded.
    	runStats = new byte[] {0x08, 0x03};
      log.info(TAG, "TensorFlow native methods already loaded");
    } catch (UnsatisfiedLinkError e1) {
      log.info(
          TAG, "TensorFlow native methods not found, attempting to load via tensorflow_inference");
      try {
        System.loadLibrary("tensorflow_inference");
        log.info(TAG, "Successfully loaded TensorFlow native methods (RunStats error may be ignored)");
      } catch (UnsatisfiedLinkError e2) {
        throw new RuntimeException(
            "Native TF methods not found; check that the correct native"
                + " libraries are present in the APK.");
      }
    }
  }

  private void loadGraph(byte[] graphDef, Graph g) throws IOException {
    final long startMs = System.currentTimeMillis();
    try {
      g.importGraphDef(graphDef);
    } catch (IllegalArgumentException e) {
      throw new IOException("Not a valid TensorFlow Graph serialization: " + e.getMessage());
    }

  
    final long endMs = System.currentTimeMillis();
    log.info(
        TAG,
        "Model load took " + (endMs - startMs) + "ms, TensorFlow version: " + TensorFlow.version());
  }

  private void addFeed(String inputName, Tensor<?> t) {
    // The string format accepted by TensorFlowInferenceInterface is node_name[:output_index].
    TensorId tid = TensorId.parse(inputName);
    runner.feed(tid.name, tid.outputIndex, t);
    feedNames.add(inputName);
    feedTensors.add(t);
  }

  private static class TensorId {
    String name;
    int outputIndex;

    // Parse output names into a TensorId.
    //
    // E.g., "foo" --> ("foo", 0), while "foo:1" --> ("foo", 1)
    public static TensorId parse(String name) {
      TensorId tid = new TensorId();
      int colonIndex = name.lastIndexOf(':');
      if (colonIndex < 0) {
        tid.outputIndex = 0;
        tid.name = name;
        return tid;
      }
      try {
        tid.outputIndex = Integer.parseInt(name.substring(colonIndex + 1));
        tid.name = name.substring(0, colonIndex);
      } catch (NumberFormatException e) {
        tid.outputIndex = 0;
        tid.name = name;
      }
      return tid;
    }
  }

  private Tensor<?> getTensor(String outputName) {
    int i = 0;
    for (String n : fetchNames) {
      if (n.equals(outputName)) {
        return fetchTensors.get(i);
      }
      ++i;
    }
    throw new RuntimeException(
        "Node '" + outputName + "' was not provided to run(), so it cannot be read");
  }

  private void closeFeeds() {
    for (Tensor<?> t : feedTensors) {
      t.close();
    }
    feedTensors.clear();
    feedNames.clear();
  }

  private void closeFetches() {
    for (Tensor<?> t : fetchTensors) {
      t.close();
    }
    fetchTensors.clear();
    fetchNames.clear();
  }

  // Immutable state.
  private final String modelName;
  private final Graph g;
  private final Session sess;

  // State reset on every call to run.
  private Session.Runner runner;
  private List<String> feedNames = new ArrayList<String>();
  private List<Tensor<?>> feedTensors = new ArrayList<Tensor<?>>();
  private List<String> fetchNames = new ArrayList<String>();
  private List<Tensor<?>> fetchTensors = new ArrayList<Tensor<?>>();

  // Mutable state.
  private byte[] runStats = new byte[] {0x08, 0x03};
}
