# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 09:52:04 2018

@author: Susmitpatil410
"""
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(
        input_image=os.path.join(
                execution_path ,"E:\\User\\Desktop\\Desktop\\flask\\objDetect\\image1.jpg"
                ), output_image_path=os.path.join(
                        execution_path ,"E:\\User\\Desktop\\Desktop\\flask\\objDetect\\image1new.jpg"
                        )
        )

for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )