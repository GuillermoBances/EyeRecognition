# -*- coding: utf-8 -*-

from Consumer import *
import argparse

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_server', type=str, default='localhost:9092', help='Kafka broker IP and port for sending results')
    parser.add_argument('--result_topic', type=str, default="capstone_drowsiness_output", help='Kafka topic for sending results of image inspection')
    parser.add_argument('--id', type=str, default="1", help='Consumer ID')
    parser.add_argument('--group', type=str, default="grupo_resultados", help='Name of consumer group')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):

    consumer = Consumer(opt.result_topic,opt.result_server,opt.id, opt.group)
    consumer.consume_results()
    
if __name__ == '__main__':

    opt = parse_opt()
    main(opt)