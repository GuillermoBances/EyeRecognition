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
        
    def consume_results(self):

        # FUNCTION USED BY FINAL CONSUMER TO RECEIVE RESULTS OF IMAGE INSPECTION FROM MULTIPLE CONSUMERS
        print("Consumer " + str(self.id) + " connected")
        num_frames = 60

        # CREATE SORTED_LIST FOR APPENDING RESULTS
        window = sorted_list(90, 1)           
        for message in self.consumer:
            # MESSAGE PUBLISH ON TOPIC
            nparr = message.value.decode()

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
            num_frames = num_frames - 1

            # MESSAGE CONTAINE CONSUMER ID, TIMESTAMP FOR SORTING RESULTS, AND RESULT FROM EYES (EMPTY, 1 OR 2 VALUES)
            id = nparr.split(";")[0]
            dt = nparr.split(";")[1]
            results = nparr.split(";")[2]
            tiempo = str(datetime.now().strftime("%Y-%m-%d %H_%M_%S_%f"))
            print("Topic receive new message from Consumer " + id + ", with timestamp " + dt + ", and results " + results + " a la hora " + tiempo)
            
            # SORTING RESULTS AND CHECKING IF ANY OPENED EYE
            window.append(nparr)