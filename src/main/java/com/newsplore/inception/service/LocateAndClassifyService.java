package com.newsplore.inception.service;

import java.util.Arrays;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.tensorflow.Operation;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import com.deepblue.tf_obj_detect.TensorFlowObjectDetectionAPIModel;
import com.newsplore.inception.service.ClassifyImageService.GraphBuilder;
import com.newsplore.inception.service.ClassifyImageService.LabelWithProbability;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class LocateAndClassifyService {
	  private final Graph inceptionGraph;
	  private final List<String> labels;
	  private float[] outputLocations;
	  private float[] outputScores;
	  private float[] outputClasses;
	  private float[] outputNumDetections;	
	  private static final int MAX_RESULTS = 100;
	  private final String[] outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
	 private  int W, H;
     private float mean, scale;
     public Session.Runner runner;
     private TensorFlowInferenceInterface inferenceInterface;
     public LocateAndClassifyService(Graph inceptionGraph, List<String> labels,
	                                @Value("${tf.image.width}") int imageW, @Value("${tf.image.height}") int imageH,
	                                @Value("${tf.image.mean}")float mean, @Value("${tf.image.scale}") float scale) {
	        this.inceptionGraph = inceptionGraph;
	        this.labels = labels;
	        this.H = imageH;
	        this.W = imageW;
	        this.mean = mean;
	        this.scale = scale;
	        final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();
	    }

	public LabelWithProbability classifyImage(byte[] imageBytes) {
		long start = System.currentTimeMillis();
		try (Tensor image = normalizedImageToTensor(imageBytes)) {
			float[] labelProbabilities = classifyAndLocate(image);
			int bestLabelIdx = maxIndex(labelProbabilities);
			LabelWithProbability labelWithProbability = new LabelWithProbability(labels.get(bestLabelIdx),
					labelProbabilities[bestLabelIdx] * 100f, System.currentTimeMillis() - start);
			log.debug(String.format("Image classification [%s %.2f%%] took %d ms", labelWithProbability.getLabel(),
					labelWithProbability.getProbability(), labelWithProbability.getElapsed()));
			return labelWithProbability;
		}
	}

	private List<Float> classifyAndLocate(Tensor image) {
	     Session s = new Session(inceptionGraph); 
	     outputLocations = new float[MAX_RESULTS * 4];
		 outputScores = new float[MAX_RESULTS];
		 outputClasses = new float[MAX_RESULTS];
		 outputNumDetections = new float[1];
		 
		 s.runner().feed("image_tensor:0", image).fetch(outputNames[0], outputLocations);
		 s.runner().feed("image_tensor:0", image).fetch(outputNames[1], outputScores);
		 s.runner().feed("image_tensor:0", image).fetch(outputNames[2], outputClasses);
		 s.runner().feed("image_tensor:0", image).fetch(outputNames[3], outputNumDetections);
		 log.info(String.format("  %d",outputLocations));
		 //Tensor result = s.runner().feed("image_tensor:0", image).fetch(outputNames).run().get(0));
		   
		return result.copyTo(new float[1][nlabels])[0];
    }

	private int maxIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	private Tensor normalizedImageToTensor(byte[] imageBytes) {
		try (Graph g = new Graph()) {
			GraphBuilder b = new GraphBuilder(g);
	            //Tutorial python here: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
	            // Some constants specific to the pre-trained model at:
	            // https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
	            //
	            // - The model was trained with images scaled to 299x299 pixels.
	            // - The colors, represented as R, G, B in 1-byte each were converted to
	            //   float using (value - Mean)/Scale.

	            // Since the graph is being constructed once per execution here, we can use a constant for the
	            // input image. If the graph were to be re-used for multiple input images, a placeholder would
	            // have been more appropriate.
	            final Output input = b.constant("input", imageBytes);
	            final Output output =
	                b.div(
	                    b.sub(
	                        b.resizeBilinear(
	                            b.expandDims(
	                                b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
	                                b.constant("make_batch", 0)),
	                            b.constant("size", new int[] {H, W})),
	                        b.constant("mean", mean)),
	                    b.constant("scale", scale));
	            try (Session s = new Session(g)) {
	                return s.runner().fetch(output.op().name()).run().get(0);
	            }
	        }
	    }

	    static class GraphBuilder {
	        GraphBuilder(Graph g) {
	            this.g = g;
	        }

	        Output div(Output x, Output y) {
	            return binaryOp("Div", x, y);
	        }

	        Output sub(Output x, Output y) {
	            return binaryOp("Sub", x, y);
	        }

	        Output resizeBilinear(Output images, Output size) {
	            return binaryOp("ResizeBilinear", images, size);
	        }

	        Output expandDims(Output input, Output dim) {
	            return binaryOp("ExpandDims", input, dim);
	        }

	        Output cast(Output value, DataType dtype) {
	            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
	        }

	        Output decodeJpeg(Output contents, long channels) {
	            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
	                .addInput(contents)
	                .setAttr("channels", channels)
	                .build()
	                .output(0);
	        }

	        Output constant(String name, Object value) {
	            try (Tensor t = Tensor.create(value)) {
	                return g.opBuilder("Const", name)
	                    .setAttr("dtype", t.dataType())
	                    .setAttr("value", t)
	                    .build()
	                    .output(0);
	            }
	        }

	        private Output binaryOp(String type, Output in1, Output in2) {
	            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
	        }

	        private Graph g;
	    }

	    @Data
	    @NoArgsConstructor @AllArgsConstructor
	    public static class LabelWithProbability {
	        private String label;
	        private float probability;
	        private long elapsed;
	    }
}
