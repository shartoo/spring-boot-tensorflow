package com.newsplore.inception.service;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DirectColorModel;
import java.awt.image.PixelGrabber;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;

import com.deepblue.tf_obj_detect.Classifier;
import com.deepblue.tf_obj_detect.TensorFlowObjectDetectionAPIModel;
import com.deepblue.tf_obj_detect.Classifier.Recognition;
import com.deepblue.tf_obj_detect.RectF;


public class TestFasterRCNN extends JPanel {
	
	 int inputWidth = 800;
	 int inputHeight = 600;
	 /* Preallocated buffers for storing image data in. */
	 // private int[] intValues = new int[inputSize * inputSize*3];
	 private static final int[] RGB_MASKS = {0xFF0000, 0xFF00, 0xFF};
	 private static final ColorModel RGB_OPAQUE =new DirectColorModel(32, RGB_MASKS[0], RGB_MASKS[1], RGB_MASKS[2]);
	 private   List<Recognition> detectResult = null;
	 private   String testImagePath = "";

     public List<Recognition> test(String testImgPath) throws IOException {
    	  String modelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\robot_inference_graph\\frozen_inference_graph.pb";
    	 //String modelFilename ="D:\\data\\model\\faster_rcnn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb";
    	  String labelFilename ="D:\\data\\robot_auto_seller\\robot_auto_seller_20171025(release)\\config\\label.txt";
        
    	 Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputHeight,inputWidth);
    	 // Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputSize,inputSize);
    	 List<Recognition> result = detector.recognizeImage(readImageBytes(testImgPath)) ;
    	 return result;
     }

	public byte[] readImageBytes(String imagePath) throws IOException {
		
		/**
		 * It's not necessary to resize image ,cause TensorFlowObjectDetectionAPIModel will
		 * 
		 * 
		 *  add by xiatao
		 */
		BufferedImage originalImage = ImageIO.read(new File(imagePath));
		/**
		 * cite https://stackoverflow.com/questions/6524196/java-get-pixel-array-from-image
		 */
		//byte[] pixels = ((DataBufferByte) originalImage.getRaster().getDataBuffer()).getData();
		int[] pixels = new int[inputWidth*inputHeight];
		/**
		 * code below refere from http://blog.csdn.net/hayre/article/details/50611591
		 */
		originalImage.getRGB(0,0, inputWidth, inputHeight, pixels, 0, inputWidth);
		System.out.println("image pixel size is:\t"+ pixels.length);
		byte[] byteValues = new byte[inputWidth * inputHeight * 3];
	    for (int i = 0; i < pixels.length; ++i) {
	    // caution about byteValues array index is 2,1,0 which can be used to change array sequence from BGR to RGB
	      byteValues[i * 3 + 2] = (byte) (pixels[i] & 0xFF);     
	      byteValues[i * 3 + 1] = (byte) ((pixels[i] >> 8) & 0xFF);
	      byteValues[i * 3 + 0] = (byte) ((pixels[i] >> 16) & 0xFF);
  }
//		PixelGrabber pg = new PixelGrabber(ImageIO.read(new File(imagePath)), 0, 0, -1, -1, true);
//		pg.grabPixels();
//		int width = pg.getWidth(), height = pg.getHeight();
//
//		DataBuffer buffer = new DataBufferInt((int[]) pg.getPixels(), pg.getWidth() * pg.getHeight());
//		WritableRaster raster = Raster.createPackedRaster(buffer, width, height, width, RGB_MASKS, null);
//		BufferedImage bi = new BufferedImage(RGB_OPAQUE, raster, false, null);
//		
//		int[] intValues = new int[inputSize * inputSize];
//		byte[] byteValues = new byte[inputSize * inputSize * 3];
//	    for (int i = 0; i < intValues.length; ++i) {
//	      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
//	      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
//	      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
//       }
		System.out.println("height : "+originalImage.getHeight());
		System.out.println("width :  "+originalImage.getWidth());
		
//		ByteArrayOutputStream baos = new ByteArrayOutputStream();
//		ImageIO.write( originalImage, "png", baos);
//		System.out.println(baos.size());
//		baos.flush();
//		byte[] imageInByte = baos.toByteArray();
		return byteValues;
	}
   
	@Override
	   public void paintComponent(Graphics g) {
	      super.paintComponent(g);  // paint background
	      //setBackground(Color.pink);
	      Font f1 = new Font("Helvetica",Font.ITALIC,24);
	      g.setFont(f1);
	      BufferedImage im;
		  try {
			System.out.println(" test image file path is:\t"+this.testImagePath);
			im = ImageIO.read(new File(this.testImagePath));
		    g.drawImage(im, 0, 0, this.inputWidth, this.inputHeight, this);
		  	for(int i = 0 ;i<this.detectResult.size() ; i++)
			{
				Recognition rec =  this.detectResult.get(i);
				if(rec.getConfidence()>0.2)
				{
					System.out.println("Title is:\t"+rec.getTitle()+ ",  confidence is:\t"+rec.getConfidence() + " , location is:\t"+ rec.getLocation() );
					int bottom = (int)rec.getLocation().bottom;
					int top = (int) rec.getLocation().top;
					int left = (int) rec.getLocation().left;
					int right = (int)rec.getLocation().right;
					int width = (right-left);
					int height = (top-bottom);
					int x =  right-width/2;
					int y = top -height/2;
					
					g.drawRect(x, y, width, height);	
					g.setColor(new Color((100+i*10)%255,(50+i*22)%255,(10+i*32)%255));			
					g.drawString(rec.getTitle(), x, y);
				}	
		   } 
		  	
		}catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	   }

	
	public static void main(String[] args) {
		TestFasterRCNN tf = new TestFasterRCNN();
		//String testImgPath =  "D:\\data\\robot_auto_seller\\robot_auto_seller_2layers\\20180205\\source_imgs\\drinking_layer0_frames_85.jpg";
		String testImgPath =  "D:\\data\\robot_auto_seller\\all_data\\images\\0_frame-0005700_o.jpg";
		tf.testImagePath =  testImgPath;
		try {
			List<Recognition> result = tf.test(testImgPath);
			tf.detectResult = result;
			 JFrame.setDefaultLookAndFeelDecorated(true);
			 JFrame frame = new JFrame("FasterRCNN detection result");
			 frame.setSize(tf.inputWidth+50,tf.inputHeight+50);
			 frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			 frame.add(tf);	 
			 frame.setVisible(true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
