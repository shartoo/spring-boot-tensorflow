package com.newsplore.inception.service;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.top.tf_obj_detect.Classifier;
import com.top.tf_obj_detect.TensorFlowObjectDetectionAPIModel;
import com.top.tf_obj_detect.Classifier.Recognition;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
@Service
@Slf4j
public class FasterRcnnService {
	 int inputWidth = 800;
	 int inputHeight = 600;
	 double threshould = 0.5;
	 /* Preallocated buffers for storing image data in. */
	 // private int[] intValues = new int[inputSize * inputSize*3];
	 private String labelFile;
	 private String modelFilename;
	 private Classifier  detector;
	
	 public FasterRcnnService(@Value("${fst.labelsPath}")String labelFile, @Value("${fst.frozenModelPath}") String modelFileName,@Value("${fst.image.width}") int inputWidth, @Value("${fst.image.height}") int inputHeight) {
		 
		this.labelFile = labelFile;
		this.modelFilename = modelFileName;
		this.inputHeight = inputHeight;
		this.inputWidth = inputWidth;
		System.out.println("faster rcnn service constructor is initial..");
		try {
			this.detector = TensorFlowObjectDetectionAPIModel.create(this.modelFilename,this.labelFile,this.inputHeight,this.inputWidth);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	 /**
	  * faster rcnn locate and classify image using image file path
	  * @param testImgPath
	  * @return
	  */
     public List<BoxLabelProbability> locateAndClassify(String testImgPath) {
    	  //String modelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\robot_inference_graph\\frozen_inference_graph.pb";
    	 //String modelFilename ="D:\\data\\model\\faster_rcnn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb";
    	  //String labelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\config\\label.txt";
        //  the parameters sequence should be inputWidth,inputHeight according to TensorFlowObjectDetectionAPIModel.but it will result into wrong prediction.
    	 // Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputSize,inputSize);
    	 List<Recognition> result = null;
    	 List<BoxLabelProbability> boxesResult = new ArrayList<BoxLabelProbability>();
		try {
			long start = System.currentTimeMillis();
			result = this.detector.recognizeImage(readImageBytes(testImgPath));
			for(int i = 0 ;i<result.size() ; i++)
			{
				Recognition rec =  result.get(i);
				if(rec.getConfidence()>this.threshould)
				{
					System.out.println("Title is:\t"+rec.getTitle()+ ",  confidence is:\t"+rec.getConfidence() + " , location is:\t"+ rec.getLocation() );				
					int bottom = (int)rec.getLocation().bottom;
					int top = (int) rec.getLocation().top;
					int left = (int) rec.getLocation().left;
					int right = (int)rec.getLocation().right;
					BoxLabelProbability boxLabelProb = new BoxLabelProbability(rec.getTitle(),rec.getConfidence(),System.currentTimeMillis()-start,top,left,bottom,right);
					boxesResult.add(boxLabelProb);
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
    	 return boxesResult;
     }
    
     public List<BoxLabelProbability> locateAndClassify(BufferedImage originalImage) {
   	  //String modelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\robot_inference_graph\\frozen_inference_graph.pb";
   	 //String modelFilename ="D:\\data\\model\\faster_rcnn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb";
   	  //String labelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\config\\label.txt";
       //  the parameters sequence should be inputWidth,inputHeight according to TensorFlowObjectDetectionAPIModel.but it will result into wrong prediction.
   	 // Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputSize,inputSize);
    	 List<Recognition> result = null;
    	 System.out.println("buffered imge size is:\t"+originalImage.getData().getDataBuffer().getSize());
    	 List<BoxLabelProbability> boxesResult = new ArrayList<BoxLabelProbability>();
		try {
			long start = System.currentTimeMillis();
			byte[] imgBytes = readImageBytes(originalImage);
			System.out.println ("image size is:  \t"+ imgBytes.length);
			result = this.detector.recognizeImage(imgBytes);
			for(int i = 0 ;i<result.size() ; i++)
			{
				Recognition rec =  result.get(i);
				if(rec.getConfidence()>this.threshould)
				{
					System.out.println("Title is:\t"+rec.getTitle()+ ",  confidence is:\t"+rec.getConfidence() + " , location is:\t"+ rec.getLocation() );				
					int bottom = (int)rec.getLocation().bottom;
					int top = (int) rec.getLocation().top;
					int left = (int) rec.getLocation().left;
					int right = (int)rec.getLocation().right;
					BoxLabelProbability boxLabelProb = new BoxLabelProbability(rec.getTitle(),rec.getConfidence(),System.currentTimeMillis()-start,top,left,bottom,right);
					boxesResult.add(boxLabelProb);
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(" locate and classify error:\t "+e.getMessage());
		}
    	 return boxesResult;
    }
     
     public byte[] readImageBytes( BufferedImage originalImage) throws IOException
     {
    	 /**
 		 * It's not necessary to resize image ,cause TensorFlowObjectDetectionAPIModel will
 		 * 
 		 * 
 		 *  add by xiatao
 		 */
 		/**
 		 * cite https://stackoverflow.com/questions/6524196/java-get-pixel-array-from-image
 		 */
 		//byte[] pixels = ((DataBufferByte) originalImage.getRaster().getDataBuffer()).getData();
 		int[] pixels = new int[inputWidth*inputHeight];
 		System.out.println(pixels.length);
 		/**
 		 * code below refere from http://blog.csdn.net/hayre/article/details/50611591
 		 */
 		for(int  i = 0; i<inputWidth;i++)
 		{
 			for(int j = 0 ;j <inputHeight; j++)
 			{
 					pixels [i*inputHeight + j] = originalImage.getRGB(i,j);
 			}
 		}
 		/**
 		 *  this method is proved not ok
 		 * originalImage.getRGB(0,0, inputWidth, inputHeight, pixels, 0, inputWidth);
 		 */
 		System.out.println("image pixel size is:\t"+ pixels.length);
 		byte[] byteValues = new byte[inputWidth * inputHeight * 3];
 	    for (int i = 0; i < pixels.length; ++i) {
 	    // caution about byteValues array index is 2,1,0 which can be used to change array sequence from BGR to RGB
 	      byteValues[i * 3 + 2] = (byte) (pixels[i] & 0xFF);     
 	      byteValues[i * 3 + 1] = (byte) ((pixels[i] >> 8) & 0xFF);
 	      byteValues[i * 3 + 0] = (byte) ((pixels[i] >> 16) & 0xFF);
   }
// 		PixelGrabber pg = new PixelGrabber(ImageIO.read(new File(imagePath)), 0, 0, -1, -1, true);
// 		pg.grabPixels();
// 		int width = pg.getWidth(), height = pg.getHeight();
 //
// 		DataBuffer buffer = new DataBufferInt((int[]) pg.getPixels(), pg.getWidth() * pg.getHeight());
// 		WritableRaster raster = Raster.createPackedRaster(buffer, width, height, width, RGB_MASKS, null);
// 		BufferedImage bi = new BufferedImage(RGB_OPAQUE, raster, false, null);
// 		
// 		int[] intValues = new int[inputSize * inputSize];
// 		byte[] byteValues = new byte[inputSize * inputSize * 3];
// 	    for (int i = 0; i < intValues.length; ++i) {
// 	      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
// 	      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
// 	      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
//        }
 		System.out.println("height : "+originalImage.getHeight());
 		System.out.println("width :  "+originalImage.getWidth());
 		
// 		ByteArrayOutputStream baos = new ByteArrayOutputStream();
// 		ImageIO.write( originalImage, "png", baos);
// 		System.out.println(baos.size());
// 		baos.flush();
// 		byte[] imageInByte = baos.toByteArray();
 		return byteValues;
     }
	public byte[] readImageBytes(String imagePath) throws IOException {
		
		 BufferedImage originalImage = ImageIO.read(new File(imagePath));
		 
		 return readImageBytes(originalImage);
	}
   
	 public static BufferedImage rotateImage(final BufferedImage bufferedimage,
	            final int degree) {
	        int w = bufferedimage.getWidth();
	        int h = bufferedimage.getHeight();
	        int type = bufferedimage.getColorModel().getTransparency();
	        BufferedImage img;
	        Graphics2D graphics2d;
	        (graphics2d = (img = new BufferedImage(w, h, type))
	                .createGraphics()).setRenderingHint(
	                RenderingHints.KEY_INTERPOLATION,
	                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
	        graphics2d.rotate(Math.toRadians(degree), w / 2, h / 2);
	        graphics2d.drawImage(bufferedimage, 0, 0, null);
	        graphics2d.dispose();
	        return img;
	    }
	 
	  	@Data
	    @NoArgsConstructor 
	    @AllArgsConstructor
	    public static class BoxLabelProbability {
	        private String title;
	        private float probability;
	        private long elapsed;
	        private int top;
	        private int left;
	        private int bottom;
	        private int right;
	    }
}
