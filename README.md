# Tensorflow-Mobile-Generic-Object-Localizer
Python Tensorflow 2 scripts for detecting objects of any class in an image without knowing their label.

![Tensorflow Generic Object Localizer](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/output.jpg)
*Original image taken from the OpenCV AI Kit - Lite, make sure to check it out: https://www.kickstarter.com/projects/opencv/opencv-ai-kit-oak-depth-camera-4k-cv-edge-object-detection*

### :exclamation::warning:The object **detector works better with images with few objects** and it starts to fail in more complex scenes. The model is suitable for automatically labelling objects for custom object detection models.

# Requirements

 * **OpenCV**, **imread-from-url** and **tensorflow**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# Tensorflow model
The original models was taken from [Tensorflow Hub](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1), download it, and place it in the **[models folder](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/tree/main/models)**. 

Use the following script to download the model:
```
python download_model.py
```

 
# Examples

 * **Image inference**:
 
 ```
 python imageObjectDetection.py 
 ```
 
 * **Webcam inference**:
 
 ```
 python webcamObjectDetection.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoObjectDetection.py
 ```

# Inference Examples
![Generic object detector figures](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/genericObjectLocalizer.gif)
 
*Original video by Animist: https://youtu.be/uKyoV0uG9rQ*

## Astronaut detection
![Astrounaut Tensorflow detection](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/astronaut.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Astronaut_Standing_On_The_Moon.png*

## Excabator detection
![Excabator Tensorflow detection](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/excavator.jpg)
 *Original image: https://en.wikipedia.org/wiki/Hitachi_Construction_Machinery_(Europe)#/media/File:ZX350LCN-3-Photo28-lo.jpg*

## Map island detection
![Map island Tensorflow detection](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/map.jpg)
 *Original image: https://ja.m.wikipedia.org/wiki/%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB:Map_of_Hawaii_highlighting_Hawaii_(island).svg*

## Phone accessories detection
![Phone accesories Tensorflow detection](https://github.com/ibaiGorordo/Tensorflow-Mobile-Generic-Object-Localizer/blob/main/docs/img/phone%20accessories.jpg)
 *Original image: https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/OnePlus_3_phone%2C_charger_and_package.jpg/1024px-OnePlus_3_phone%2C_charger_and_package.jpg*

## And many more

# References:
* Original model: https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1
