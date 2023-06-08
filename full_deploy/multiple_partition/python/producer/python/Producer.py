# -*- coding: utf-8 -*-

import cv2
from kafka import KafkaProducer

#
class Producer():

    def __init__(self, topic,servers,id):

        # CREATE KAFKA PRODUCER
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=servers, api_version=(0,10,2))
        self.id = id

    def send_camera(self,frame):

        # SEND ENCODED IMAGE TO TOPIC
        ret, buffer = cv2.imencode('.jpg', frame)
        self.producer.send(self.topic, buffer.tobytes())
        self.producer.flush()
        print("Producer " + str(self.id) + " sent image successfully")

    def send_text(self,text):

        # SEND ENCODED TEXT TO TOPIC (USED FOR TESTING)
        print("Producer " + str(self.id) + " sending this text: " + text)
        buffer = str.encode(text)
        self.producer.send(self.topic, buffer)
        self.producer.flush()
        print("Producer " + str(self.id) + " sent text successfully")