# image classify and faster-rcnn detection


## Prerequisites
- JDK 8
- nothing else

## Run

On windows, enter the directory of project and type 'cmd' in the blank nevigate .Then type run command 

```
gradle bootrun
```

Navigate to http://localhost:8080 and upload an image. The backend will categorize the image and output the result

### Run classify

Just clik on the botton `classify` ,the result will show like this


<div align="center" style="text-align:center"><img src="cat_classified.jpg" width="360"/></div>

### Run locate and classify

There are some problems for showing the locate and classify result,python code to call the function is  below.

```
import requests

files={'app_id':(None,'123456'),
    'version':(None,'2256'),
    'platform':(None,'linux'),
    'file':open('/home/tao/workspace/imgs/0_frame-0009417_o.jpg','rb')
 }
url = "http://localhost:8080/api/locateAndClassify"
response=requests.post(url,files=files)
print("=======  predict result ========")
print(response.content)
```


## TODO

The current version is a **CPU version**, a **Windows GPU Version is not ready till now**. According to tensorflow official words, a tensorflow model cannot be read by Java till version 1.7.

## cites

Head to [the blog post](https://blog.newsplore.com/2017/07/31/zero-to-image-recognition-in-60-seconds-with-tensorflow-and-spring-boot) for the detail about how to do image classify.

## thanks

Thanks to [@mcjojos](https://github.com/mcjojos), a port to Kotlin is now [available](https://github.com/florind/inception-serving-sb/tree/kotlin) in a separate branch. Woot!

