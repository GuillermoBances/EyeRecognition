# CV_Capstone

Capstone project for drownsiness detection with real-time webcam images. Current version in full_deploy folder.

## RUN 

1) Install requirements.txt. Pytorch packages are commented because they have to be installed via an expecific command.

2) For using GPU for inference on Windows, install in your computer CUDA Toolkit 11.4 (V11.4.48, cuda_11.4.r11.4/compiler.30033411_0), download cuDNN v.8.4.0 library packages and included them into Nvidia GPU Toolkit directory, and install Pytorch for CUDA 11.3 from https://pytorch.org/get-started/locally/ (at the time of writing this, command is as follows: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113). Finally, include into your PATH environment variable the root to Nvidia CUDA Toolkit bin and libnvvp folders. 

3) Install Docker and docker-compose for deploy containers with Kafka and Zookeeper images.

4) Run main.py. This script will clean all docker containers, images, networks and volumes. Then it will run docker containers with docker-compose.yml file for Zookeeper and Kafka, and python scripts for consumer_main (inference from images published on Kafka topic, using YOLO model currently on production and saved on Azure Blob Storage), consumer_results (receive inference results from topic) and producer (publish images on Kafka topic). Note: in case of Exception "Install pypiwin32 package to enable npipe:// support", check https://github.com/twosixlabs/armory/issues/156 and run "python <path-to-python-env>\Scripts\pywin32_postinstall.py -install".

5) If you want to do it manually, follow this steps:

    a) Run Docker Daemon/ Docker Desktop.
    b) Execute this comand for running docker containers with Kafka and Zookeeper images: docker-compose -f docker-compose.yml up
    c) Once Kafka is running, run python/consumer_main/python/consumer_main.py and python/consumer_results/python/consumer_results.py.
    d) Run python/producer/python/producer_main.py for capturing images from webcam.
    e) Results must be received on consumer_results at same rate as fps are captured from webcam.

## INFERENCE AND METRICS

1) Inside python folder there are 2 scripts for inference and mAP metrics for any dataset provided. 2 datasets are included, with images of a public dataset of famous people faces, and images from our webcam, with YOLO labels for eyes (class, X, Y, width and height) and results from inference included, and 2 weights from trained model versions, yolo_v2 (current best model, adn the one used in production) and yolo_v3 (most recent model, but with worse results than current).

2) Run inference.py for prediction in a dataset. Images must be inside a folder named "images" inside a folder that can be passed as argument (default directory is "data"). Inference results will be saved on a folder named "results". Results from inference will be saved on Azure Table Storage, as yolo weights used for inference, so this weights can be downloaded and used for webcam images.