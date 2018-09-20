package com.newsplore.inception.api;

import com.top.tf_obj_detect.Classifier;
import com.top.tf_obj_detect.TensorFlowObjectDetectionAPIModel;
import com.top.tf_obj_detect.Classifier.Recognition;
import com.newsplore.inception.service.ClassifyImageService;
import com.newsplore.inception.service.FasterRcnnService;
import com.newsplore.inception.service.FasterRcnnService.BoxLabelProbability;

import net.sf.jmimemagic.Magic;
import net.sf.jmimemagic.MagicMatch;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

@RestController
@RequestMapping("/api")
public class AppController {

    private ClassifyImageService classifyImageService;
    private FasterRcnnService fasterRcnnService;

    public AppController(ClassifyImageService classifyImageService,FasterRcnnService fasterRcnnService) {
        this.classifyImageService = classifyImageService;
        this.fasterRcnnService =  fasterRcnnService;
    }

    @PostMapping(value = "/classify")
    @CrossOrigin(origins = "*")
    public ClassifyImageService.LabelWithProbability classifyImage(@RequestParam MultipartFile file) throws IOException {
        checkImageContents(file);
        return classifyImageService.classifyImage(file.getBytes());
    }
    
    @PostMapping(value = "/locateAndClassify")
    @CrossOrigin(origins = "*")
    public String fasterrcnn(@RequestParam MultipartFile file) throws IOException {
        checkImageContents(file);
   	    // Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputSize,inputSize);
        InputStream in = new ByteArrayInputStream(file.getBytes());
        System.out.println( " the image bytes read total:\t "+ in.available());
        List<BoxLabelProbability> result = fasterRcnnService.locateAndClassify(ImageIO.read(in));
        String resStr = "";
        HashMap<String,Integer>  resultMap = new HashMap<String,Integer>();
        for(BoxLabelProbability box:result )
        {
        	if(resultMap.containsKey(box.getTitle()))
        	{
        		int value = resultMap.get(box.getTitle()) + 1;
        		resultMap.replace(box.getTitle(), value);
        	}
        	else
        	{
        		resultMap.put(box.getTitle(),1);
        	}
        }
        Iterator<Entry<String, Integer>> iter = resultMap.entrySet().iterator();
       while (iter.hasNext()) 
       {
    	   HashMap.Entry<String,Integer> entry = (Map.Entry<String,Integer>) iter.next();
    	   String key = entry.getKey();
    	   int  val = entry.getValue();
    	   resStr =  resStr + "," + key + "  :  " + String.valueOf(val); 
      }
       resStr =  resStr.substring(1,resStr.length());
       System.out.println("  The FasterRCNN Detect result is:\t"+resStr);
        return resStr;
    }
    @PostMapping(value = "/locateAndClassifyDetail")
    @CrossOrigin(origins = "*")
    public List<FasterRcnnService.BoxLabelProbability> fasterrcnnDetail(@RequestParam MultipartFile file) throws IOException {
        checkImageContents(file);
   	    // Classifier detector = TensorFlowObjectDetectionAPIModel.create(modelFilename,labelFilename,inputSize,inputSize);
        InputStream in = new ByteArrayInputStream(file.getBytes());
        List<BoxLabelProbability> result = fasterRcnnService.locateAndClassify(ImageIO.read(in));
        return result;
    }
   
    @RequestMapping(value = "/")
    public String index() {
        return "index";
    }

    private void checkImageContents(MultipartFile file) {
        MagicMatch match;
        try {
            match = Magic.getMagicMatch(file.getBytes());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        String mimeType = match.getMimeType();
        if (!mimeType.startsWith("image")) {
            throw new IllegalArgumentException("Not an image type: " + mimeType);
        }
    }

}
