import os
import ctypes
import time
import docker

from typing import Optional
import subprocess


def is_container_running(container_name: str) -> Optional[bool]:
    
    while(True):
        
        try:
            cli = docker.APIClient()
            inspect_dict = cli.inspect_container('capstone_drowsiness_kafka_server')
            state = inspect_dict['State']
            
            is_running = state['Status'] == 'running'
            if is_running:
                time.sleep(30)
                break
        except Exception as e:

            message = str(e)
            code = message.split(" ")
            if code[0]=="404":
                "Waiting for container to start running"
            else:
                print(e)
            time.sleep(5)

def out_header(string):
    print(string)

def clean_up(string):

    ctypes.windll.kernel32.SetConsoleTitleW("Capstone_Drowsiness")   

    out_header('Cleanup')
    output = subprocess.check_output("docker ps -q", shell=True)
    if len(output) > 0:
        os.system('docker kill $(docker ps -q)')
    os.system('docker system prune')
    os.system('docker container prune')
    output = subprocess.check_output("docker image ls -q", shell=True)
    if len(output) > 0:
        os.system('docker image prune')
    output = subprocess.check_output("docker volume ls -q", shell=True)
    if len(output) > 0:
        os.system('docker volume prune')
    out_header(string)

if __name__ == "__main__":

    #    

    launch_title = 'Start up'
    kafka_server_container_name = 'capstone_drowsiness_kafka_server'

    #

    clean_up(launch_title)
    
    # Kafka-Server #
    os.system('start "Docker Kafka Server" cmd /k "docker-compose -f docker-compose.yml up"')
    
    is_container_running(kafka_server_container_name)
    
    out_header('Kafka Server Running')
    
    # Kafka-Consumer #
    
    out_header('Running Consumers for Inference and Results')

    os.system('start "Kafka Consumer for Inference" cmd /k "python python/consumer_main/python/consumer_main.py --main True"')
    os.system('start "Kafka Consumer for results" cmd /k "python python/consumer_results/python/consumer_results.py"')
    #os.system('start "Docker Consumer" cmd /k "docker build -t consumer python/consumer/. && docker run -it consumer"')
    
    time.sleep(45)
    # Kafka-Producer #
    #os.system('start "Docker Producer" cmd /k "docker build -t producer python/producer/. && docker run -it producer"')
    out_header('Running Producer for capturing webcam images and publishing on Kafka Topic')
    os.system('start "Kafka Producer" cmd /k "python python/producer/python/producer_main.py --main True"')