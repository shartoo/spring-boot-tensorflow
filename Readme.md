# image classify and faster-rcnn detection

[![Build Status](https://travis-ci.org/florind/inception-serving-sb.svg?branch=master)](https://travis-ci.org/florind/inception-serving-sb)
[![Coverage Status](https://coveralls.io/repos/github/florind/inception-serving-sb/badge.svg?branch=master)](https://coveralls.io/github/florind/inception-serving-sb?branch=master)

TL;DR: for the impatient ones wanting to have a web service for image recognition without any Tensorflow prerequisites: run ```./gradlew fetchInceptionFrozenModel bootrun```, navigate to http://localhost:8080 and upload an image. The backend will categorize the image and output the result along with the probability.

Screenshot (non clickable)<br/>
<div align="center" style="text-align:center"><img src="cat_classified.jpg" width="560"/></div>
  

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

Just clik on the botton `classify` ,the result will show

### Run locate and classify


## TODO

The current version is a **CPU version**, a **Windows GPU Version is not ready till now**. According to tensorflow official words, a tensorflow model cannot be read by Java till version 1.7.

## cites

Head to [the blog post](https://blog.newsplore.com/2017/07/31/zero-to-image-recognition-in-60-seconds-with-tensorflow-and-spring-boot) for the detail about how to do image classify.

## thanks

Thanks to [@mcjojos](https://github.com/mcjojos), a port to Kotlin is now [available](https://github.com/florind/inception-serving-sb/tree/kotlin) in a separate branch. Woot!

