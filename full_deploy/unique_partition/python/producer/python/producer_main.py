# -*- coding: utf-8 -*-

from Producer import *
import numpy as np
import pandas as pd
import cv2
import time
from datetime import datetime
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default="capstone_drowsiness_intake" , help='Kafka topic where images are being sent')
    parser.add_argument('--server', type=str, default='localhost:9092', help='Kafka broker IP and port for retrieving images')
    parser.add_argument('--id', type=str, default="1", help='Producer ID')
    parser.add_argument('--images', type=int, default=480, help='Number of images from webcam to send to broker')
    parser.add_argument('--save_images', type=str2bool, default=False, help='Flag to save original images to local')
    parser.add_argument('--main', type=str2bool, default=False, help='Flag to save images on right directory if script is runned via main.py') 
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):

    # KAFKA PRODUCER IS CREATED WITH SERVER, TOPIC AND ID PASSED AS ARGS
    server = opt.server    
    topic = opt.topic
    id = opt.id
    producer = Producer(topic,server,id)

    print("Publishing feed!")

    # WEBCAM 
    camera = cv2.VideoCapture(0)
    num_frames = 60

    # TOTAL NUMBER OF IMAGES TO SEND FROM PRODUCER
    total_frames = opt.images
    try:
        while(total_frames > 0):

            # IMAGE READ FROM WEBCAM
            success, frame = camera.read()

            # TIME CALCULATION FOR ESTIMATE FPS
            if num_frames == 60:
                start = time.time()
            elif num_frames == 0:
                num_frames = 61
                end = time.time()
                seconds = end - start
                print ("Time taken : {0} seconds".format(seconds))
                # Calculate frames per second
                fps  = 60 / seconds
                print("Estimated frames per second : {0}".format(fps))

            # TIMESTAMP INCLUDED ON IMAGE
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            frame = cv2.putText(frame, dt, (0,20), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1.0, color = (255,0,0), thickness = 1, lineType = cv2.LINE_AA)
            producer.send_camera(frame)
            num_frames = num_frames - 1  
            total_frames = total_frames - 1  

            # OPTIONAL: SAVE ORIGINAL IMAGES SENT FROM PRODUCER IN LOCAL FILE TO COMPARE WITH ONES RECEIVED AT CONSUMERS 
            if opt.save_images:
                filename = dt + ".jpg"
                filename = filename.replace(":","_")

                if opt.main:
                    actual_directory = os.path.join(os.getcwd(),'python','producer','python')
                else:
                    actual_directory = os.getcwd()

                save_path = os.path.join(actual_directory,"images")
                if os.path.isdir(save_path) == False:
                   os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path,filename),frame)

    except Exception as e:
        print(e)
        print("\nExiting.")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    opt = parse_opt()
    main(opt)