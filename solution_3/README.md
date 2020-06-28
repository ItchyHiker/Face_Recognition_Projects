1. 使用MTCNN检测人脸和5点关键点
2. 决定要裁剪的人脸大小
3. 使用参考5点关键点点位对检测到的人脸进行校正
    ```Python
    # reference facial points, a list of coordinates (x,y)
    REFERENCE_FACIAL_POINTS = [
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ]
    DEFAULT_CROP_SIZE = (96, 112)
    ```
4. 阅读lfw数据的[README](http://vis-www.cs.umass.edu/lfw/README.txt)信息
下载 [pairs.txt](http://vis-www.cs.umass.edu/lfw/pairs.txt)文件，里面有3000对match人脸图像和3000对unmatch人脸图像

5. 裁剪CASIA-WebFace和LFW数据库

