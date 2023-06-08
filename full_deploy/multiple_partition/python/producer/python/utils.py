import numpy as np
import pandas as pd
import os
import cv2

def send_camera(frame,producer):
    ret, buffer = cv2.imencode('.jpg', frame)
    producer.send(producer.topic, buffer.tobytes())
    producer.flush()
    print("Producer " + str(producer.id) + "sent image successfully")

def send_text(text,producer):
    print("Producer " + str(producer.id) + "sending this text: " + text)
    buffer = str.encode(text)
    producer.send(producer.topic, buffer)
    producer.flush()
    print("Producer " + str(producer.id) + "sent text successfully")