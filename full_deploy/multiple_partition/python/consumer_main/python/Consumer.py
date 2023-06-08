# -*- coding: utf-8 -*-

import cv2
import numpy as np
from kafka import  KafkaConsumer, KafkaProducer
import os
import pandas as pd
import time
import torch
import shutil
from util import *
from datetime import datetime

# FUNCTION FOR START ALERT AFTER 90 IMAGES WITH CLOSED EYES

class Consumer():
    
    def __init__(self, topic, broker, id, group):

        # CREATE KAFKA CONSUMER
        self.consumer = KafkaConsumer(topic, bootstrap_servers=broker, consumer_timeout_ms=300000, group_id = group, api_version=(0,10,2), auto_offset_reset='earliest')
        self.id = id
        self.server = broker

    def consume_camera(self, result_broker, result_topic, remote, save_images, main, show):   

        # FUNCTION USED BY CONSUMERS THAT WILL USED YOLO MODEL TO INSPECT IMAGE
        # IF REMOTE FLAG, DOWNLOAD YOLO MODEL AND WEIGHTS FROM AZURE BLOB STORAGE. ELSE, YOLO MUST BE ALREADY DOWNLOADED ON 
        # MODEL DIRECTORY
        
        if main:
            actual_directory = os.path.join(os.getcwd(),'python','consumer_main','python')
        else:
            actual_directory = os.getcwd()

        yolo_path = os.path.join(actual_directory,'model','yolo')
        yolo_directory = os.path.join(yolo_path,'ultralytics_yolov5_master')
        directory_exist = os.path.isdir(yolo_path)

        if remote or directory_exist == False:
            if directory_exist == True:
                shutil.rmtree(yolo_path)
            os.makedirs(yolo_path)
            current_model_in_production = download_from_azure(yolo_path)
            yolo_model_name = os.path.join(yolo_path,current_model_in_production)

        else:
            files = os.listdir(yolo_path)
            for file in files:
                if ".pt" in file:
                    yolo_model_name = os.path.join(yolo_path,file)
                    break

        try:
            # CREATE TORCH MODEL WITH YOLO ARCHITECTURE AND PRETRAINED WEIGHTS
            yolo_model = torch.hub.load(yolo_directory, 'custom', path=yolo_model_name, source = "local",force_reload=True)
            print('Consuming web_cam')
            num_frames = 60

            # CREATE KAFKA PRODUCER FOR SENDING FINAL RESULTS
            producer = KafkaProducer(bootstrap_servers=result_broker, api_version=(0,10,2))

            exit = 0
            # CONSUME MESSAGES ON TOPIC
            print("Consumer ready for receiving images")
            for message in self.consumer:
                # DECODE MESSAGE TO OBTAIN IMAGE
                nparr = np.frombuffer(message.value, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # EXTRACT TIMESTAMP FROM MESSAGE GIVEN AT PRODUCER SIDE FOR SORTING RESULTS 
                dt = str(message.timestamp)

                # INSPECT IMAGE WITH YOLO MODEL
                result = yolo_model(frame)        
                df = result.pandas().xyxy[0]
                result_eyes = []

                # TIME CALCULATION FOR ESTIMATE FPS
                if num_frames == 60:
                    start = time.time()
                elif num_frames == 0:
                    num_frames = 61
                    end = time.time()
                    seconds = end - start
                    print ("Time taken : {0} seconds".format(seconds))
                    fps  = 60 / seconds
                    print("Estimated frames per second : {0}".format(fps))

                # CHECKING RESULTS FROM ALL OBJECT RECOGNISED FOR IMAGE
                for j in range(len(df)):
                    # OBJECT CONFIDENCE MUST BE GREATER THAN 0.5 TO BE EVEN CONSIDER AS AN EYE
                    if float(df["confidence"][j]) > 0.5:
                        # OBJECT RECOGNITION RESULTS: XMIN, XMAX, YMIN, YMAX AND CLASS
                        xmin = int(df["xmin"][j])
                        xmax = int(df["xmax"][j])
                        ymin = int(df["ymin"][j])
                        ymax = int(df["ymax"][j])
                        clas = int(df["class"][j])
                        # YOLO MODEL TRAIN FOR 2 CLASSES: 0 FOR OPENED EYES AND 1 FOR CLOSED EYES
                        result = "OPENED" if clas == 0 else "CLOSED"
                        result_eyes.append(result)
                        if save_images:
                            #DRAW BOXES AND TEXT OVER EYES FOR IDENTIFYING RESULTS ON IMAGES
                            text_x = int(xmin)
                            text_y = int(ymin-20)
                            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color=(255, 0, 0), thickness=3)
                            cv2.putText(frame, result, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, cv2.LINE_AA)
                
                # CREATE MESSAGE FOR RESULT TOPIC, WITH CONSUMER ID, TIMESTAMP AND RESULTS
                text = str(self.id) 
                data = text + ";" + dt + ";" + str(result_eyes)
                buffer = str.encode(data)
                print("Consumer "+ text + " has inspected image with timestamp " + dt + " and obtain this results: " + str(result_eyes))
                
                # SEND RESULTS TO TOPIC
                producer.send(result_topic, buffer)
                producer.flush()
                num_frames = num_frames - 1
                
                # IF FLAG, SAVING IMAGES ON LOCAL
                if save_images:
                    #filename = dt + ".jpg"
                    filename = dt + '_' + str(datetime.now().strftime("%Y-%m-%d %H_%M_%S_%f")) + ".jpg"
                    save_path = os.path.join(actual_directory,"images")
                    if os.path.isdir(save_path) == False:
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,filename),frame)
                
                if show:
                    cv2.imshow("Webcam with YOLO detection",frame)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        exit = 1
                        cv2.destroyAllWindows()
                if exit ==1:
                    break
            if show:
                cv2.destroyAllWindows()
        except Exception as e:
            print('#############################')
            print(e)
            print('#############################')
            exit(1)
                    
