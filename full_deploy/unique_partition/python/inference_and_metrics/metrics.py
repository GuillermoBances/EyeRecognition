# -*- coding: utf-8 -*-

# SCRIPT FOR CALCULATING MAP, PRECISION, RECALL AND CONFUSION MATRIX METRICS FOR YOLO RESULTS

import cv2
import numpy as np
import os
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mean
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, TableClient, UpdateMode
import time
from tqdm import tqdm

# FUNCTION TO CALCULATE INTERSECTION OVER UNION OF 2 BOXES
def calculate_IoU(box1,box2):
    xmin1, xmax1, ymin1, ymax1 = box1[0], box1[1], box1[2], box1[3]
    xmin2, xmax2, ymin2, ymax2 = box2[0], box2[1], box2[2], box2[3]

    xside = max(0,min(xmax1,xmax2) - max(xmin1,xmin2))
    yside = max(0,min (ymax1,ymax2) - max(ymin1,ymin2))

    interArea = xside*yside

    box1area = (xmax1-xmin1)*(ymax1-ymin1)
    box2area = (xmax2-xmin2)*(ymax2-ymin2)

    iou = interArea / float(box1area + box2area - interArea)
    return iou

# FUNCTION TO EXTRACT X, Y AND CLASS FROM YOLO LABEL
def yolo2box(line,width,height):
    token = line.split(" ")
    xmin, xmax = (float(token[1]) - (float(token[3])/2)) * width , (float(token[1]) + (float(token[3])/2)) * width
    ymin, ymax = (float(token[2]) - (float(token[4])/2)) * height , (float(token[2]) + (float(token[4])/2)) * height
    clas = int(token[0])
    return xmin, xmax, ymin, ymax, clas

# FUNCTION TO EXTRACT X, Y AND CLASS FROM RESULTS LABEL
def inference2box(line):
    token = line.split(", ")
    xmin, xmax = float(token[0]) , float(token[2])
    ymin, ymax = float(token[1]) , float(token[3])
    confidence = float(token[4])
    clas = int(token[5])
    return xmin, xmax, ymin, ymax, confidence, clas

def calculate_mAP(opt):
    # MINIMUN IOU TO CONSIDER A BOX FOR MAP 
    iou_threshold = opt.threshold
    print("Measuring metrics for images saved in " + opt.root)
    # DIRECTORIES WHERE IMAGES, YOLO LABELS AND RESULTS ARE SAVED
    path = os.path.join(opt.root,"images")
    test_path = os.path.join(opt.root,"labels")
    label_path = os.path.join(opt.root,"results")

    # DIRECTORY WHERE IMAGES WITH BOXES WILL BE SAVED
    save_box = os.path.join(opt.root,"boxes")
    if os.path.isdir(save_box) == False and opt.save_images:
        os.makedirs(save_box)

    files = os.listdir(path)
    total_number_frames = len(files)
    total_area = []
    total_precision = []
    total_recall = []
    total_f1 = []
    classes = [0,1]

    # CALCULATION OF MAP FOR EACH CLASS
    for element in classes:
        mAP_table = []
        total_boxes = 0
        y = []
        prediction = []
        for i in tqdm(range(total_number_frames), desc = "mAP calculations for class " + str(element)):
            # READ IMAGE AND ITS DIMENSIONS FOR CALCULATE GLOBAL POSITION OF EYES FROM YOLO LABELS AND RESULTS
            image = files[i]
            frame = cv2.imread(os.path.join(path,image))
            height, width = frame.shape[0], frame.shape[1]
            # READ YOLO LABEL FOR CURRENT IMAGE TO SAVE REAL EYES POSITIONS
            test_name = image.split(".")[0] + ".txt"
            txt_file = os.path.join(test_path,test_name)
            f = open(txt_file, 'r') 
            lines = f.readlines()
            true_box = []
            for line in lines:
                # TRANSLATE YOLO LABEL TO X AND Y COORDINATES
                xmin, xmax, ymin, ymax, clas = yolo2box(line, width, height)
                true_box.append([xmin, xmax, ymin, ymax, clas])
                if clas == element:
                    total_boxes += 1
                # IF 1º CLASS AND SAVE_IMAGES FLAG IS TRUE, SAVE IMAGES WITH BOXES
                if element == 0 and opt.save_images:
                    cv2.rectangle(frame, (int(xmin),int(ymin)) , (int(xmax),int(ymax)), (0,255,0), 2)
                    text_y = int(ymin)-15
                    cv2.putText(frame, str(clas), (int(xmin), int(text_y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
            # CLOSE TXT OBJECT
            f.close()
            
            # READ INFERENCE RESULTS FOR CURRENT IMAGE
            name = image.split(".")[0] + ".txt"
            txt_file = os.path.join(label_path,name)
            f = open(txt_file, 'r') 
            lines = f.readlines()
            index = 0
            for line in lines:
                # FIRST ROW IS COLUMN NAMES
                if index > 0:
                    # TRANSLATE INFERENCE RESULTS TO X AND Y COORDINATES
                    xmin, xmax, ymin, ymax, confidence, clas = inference2box(line)
                    detected_box = [xmin, xmax, ymin, ymax]
                    # IF, AS CONFIDENCE IS BIG, THIS OBJECT IS CONSIDER AN EYE
                    if confidence > opt.confidence:
                        # IF 1º CLASS AND SAVE_IMAGES FLAG IS TRUE, SAVE IMAGES WITH BOXES
                        if element == 0 and opt.save_images:
                            cv2.rectangle(frame, (int(xmin),int(ymin)) , (int(xmax),int(ymax)), (0,0,255), 2)
                            text_y = int(ymin)-15
                            cv2.putText(frame, str(clas), (int(xmin), int(text_y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
                        # CALCULATE WHAT BOX HAS BIGGER IOU, AND CONSIDER THIS FOR MAP
                        map_status = 0
                        detected_status = 0
                        max_iou = 0
                        # IF INFERENCE CLASS IS CURRENT CLASS, IT CAN BE CONSIDER FOR MAP. IF NOT, ONLY IS CONSIDER FOR CLASIFICATION REPORT AND CONFUSION MATRIX
                        if clas == element:
                            # CHECK IF INFERENCE BOX HAS IOU BIGGER THAN THRESHOLD FOR ALL REAL BOXES IN THIS IMAGE, AND IF THE INFERENCE CLASS IS THE SAME AS REAL CLASS
                            for box in true_box:
                                iou = calculate_IoU(box,detected_box)
                                if iou >= max_iou and iou > iou_threshold:
                                    detected_status = 1
                                    true_box_clas = box[4]
                                    clas_predicted = clas
                                    max_iou = iou
                                    # IF INFERENCE AND REAL CLASSES ARE SAME, ITS A TRUE POSITIVE. ELSE ITS A FALSE POSITIVE
                                    if true_box_clas == clas_predicted:
                                        map_status = 1
                                    else:
                                        map_status = 0
                            # IF INFERENCE BOX HAS BIGGER IOU WITH ANY BOX AND CLASSES ARE SAME, ITS A TRUE POSITIVE. ELSE ITS FALSE POSITIVE
                            if map_status == 1:
                                mAP_table.append([image,confidence,max_iou,1])
                            else:
                                mAP_table.append([image,confidence,max_iou,0])
                        # IF INFERENCE CLASS IS NOT CURRENT CLASS, THIS INFERENCE BOX CAN ONLY IS CONSIDER FOR CLASIFICATION REPORT AND CONFUSION MATRIX, AND ONLY IF HAS BIGGER IOU WITH ANY REAL BOX
                        else:
                            for box in true_box:
                                iou = calculate_IoU(box,detected_box)
                                if iou > max_iou and iou > iou_threshold:
                                    detected_status = 1
                                    true_box_clas = box[4]
                                    clas_predicted = clas
                                    max_iou = iou
                        if detected_status == 1:
                            y.append(true_box_clas)
                            prediction.append(clas_predicted)
                index += 1
            f.close()
            # IF 1º CLASS AND SAVE_IMAGES FLAG IS TRUE, SAVE IMAGES WITH BOXES
            if element == 0 and opt.save_images:
                cv2.imwrite(os.path.join(save_box,image),frame)

        # ONCE ALL IMAGES AND BOXES HAD BEEN CHECKED, MAP ITS CALCULATED WITH AUC OF RECALL (X-AXIS) VS PRECISION (Y-AXIS) FOR CURRENT CLASS
        precision = []
        recall = []
        positives = 0
        detected = 0

        # CHECKING ALL INFERENCE BOXES RESULTS
        for i in range(len(mAP_table)):
            # CURRENT NUMBER OF INFERENCE BOXES CHECKED
            positives += 1
            # IF ITS A TRUE POSITIVE, DETECTED +1. IF FALSE POSITIVE, NO CHANGE
            if mAP_table[i][3]==1:
                detected += 1
            # PRECISION = CURRENT NUMBER OF TRUE POSITIVES / CURRENT NUMBER INFERENCE BOXES CHECKED (ITS INCREASING OR DECREASING, DEPENDING ON WHETHER ACTUAL INFERENCE BOX IS TRUE POSITIVE OR FALSE POSITIVE)
            precision.append(detected/positives)
            # RECALL = CURRENT NUMBER OF TRUE POSITIVES / TOTAL BOXES (ITS CONTINUOUSLY INCREASING, AS TOTAL BOXES IS ALWAYS THE SAME VALUE)
            recall.append(detected/total_boxes)

        # MAP IS CALCULATED AS AUC OF RECALL (X-AXIS) VS PRECISION (Y-AXIS)
        area = auc(recall, precision)
        # FINAL PRECISION AND RECALL ARE LAST ONES CALCULATED
        final_precision = precision[-1]
        final_recall = recall[-1]
        f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)
        print("Estimated precision score for class {0}: {1}".format(element,final_precision))
        print("Estimated recall score for class {0}: {1}".format(element,final_recall))
        print("Estimated F1 score for class {0}: {1}".format(element,f1))
        print("Estimated mAP for class {0}: {1}".format(element,area))
        total_precision.append(final_precision)
        total_recall.append(final_recall)
        total_f1.append(f1)
        total_area.append(area)
        # PLOT RECALL VS PRECISION AND SAVE FIGURE
        if opt.save_images:
            plt.figure(figsize=(18, 18))
            plt.plot(recall,precision)
            plt.title("mAP for model and class " + str(element) + ": " + str(area))
            plt.savefig('mAP_model_IoU_' + str(opt.threshold)+'_and_class_' + str(element) + '.png')

    total_map = mean(total_area)
    print("Estimated mAP for model: {0}".format(total_map))
    # WITH ALL INFERENCE BOXES THAT HAS A BIGGER IOU THAN THRESHOLD FOR ANY REAL BOX, AND THE REAL CLASS FOR THAT BOX, A CONFUSION MATRIX CAN BE CALCULATED. 
    # TAKE INTO ACCOUNT THAT THIS REPORT ONLY CONSIDER THOSE BOXES THAT HAS IOU BIGGER THAN THRESHOLD, SO MANY REAL AND INFERENCE BOXES ARE NOT CONSIDER IN THIS REPORT.
    # THIS METRIC WILL MEASURE, FROM THOSE EYES RIGHTLY DETECTED, HOW GOOD IS AT CLASSIFING AS OPENED OR CLOSED.
    print(classification_report(y, prediction))
    print(confusion_matrix(y, prediction))
    confusion = confusion_matrix(y, prediction).tolist()

    return total_area, total_recall, total_precision, total_f1, confusion

def create_azure_connection():

    # CREATE CONNECTION CLIENTS WITH AZURE BLOB STORAGE AND AZURE TABLE STORAGE WITH CONNECTION STRING
    connect_str = "DefaultEndpointsProtocol=https;AccountName=d2mcapstones00;AccountKey=cHCeAzdLoZKhZi6d/wIzeDJMQJeWF3OsnbcuDgZtskgnsmEXYezGGnSBGGWd+J9hsyVIfDddjtXP+AStPlvNjA==;EndpointSuffix=core.windows.net"
    
    # CHECK ALL CONTAINERS IN AZURE BLOB STORAGE AND SELECT DROWSINESS CONTAINER
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    test_containers = blob_service_client.list_containers()
    for element in test_containers:
        if "drowsiness" in element.name:
            container = element

    # CHECK ALL TABLES IN AZURE TABLE STORAGE AND SELECT TABLE FOR CHECKING CURRENT MODEL IN PRODUCTION 
    table_service_client = TableServiceClient.from_connection_string(connect_str)
    list_tables = table_service_client.list_tables()
    for element in list_tables:
        if "drowsinessModel" in element.name:
            table = element

    tableClient = TableClient.from_connection_string(connect_str, table_name=table.name)

    return blob_service_client, container, tableClient

def check_model_on_production(opt, blob_service_client, container, tableClient, model_info):
    # LIST ALL ENTITIES IN TABLE AND CHECK FOR CURRENT MODEL IN PRODUCTION
    entities = list(tableClient.list_entities())

    # CHECK IF THERE IS ANY ENTITY IN TABLE
    if entities:
        for entity in entities:
            status = entity["status"]
            if status == "in_production":
                current_model = entity
                break
    
        print("Current model in production: " + current_model["PartitionKey"])
        mAP = float(model_info['model_metrics']['Total_mAP'])
        result = float(current_model["Total_mAP"])
        print("Current mAP: {}".format(mAP))
        print("mAP from model in production: {}".format(result))
        fps = round(float(model_info['model_metrics']['FPS']),2)

        if result > mAP or fps < 30:
            # NO CHANGE NEEDED ON CURRENT MODEL
            model_info['status'] = "none"
            print("Model not improves metrics for current model in production")
        else:
            # MODIFY STATUS IN CURRENT MODEL ENTITY TO NONE
            current_model['status'] = "none"
            aux = tableClient.update_entity(mode=UpdateMode.REPLACE, entity=current_model)
            model_info['status'] = "in_production"
            print("Uploading this model as current in production")
    # IF NO ENTITIES, NO CHECK IS NEEDED AND CURRENT MODEL IS SAVED INTO PRODUCTION
    else:
        print("No model into production yet")
        model_info['status'] = "in_production"
    
    # CREATE NEW ENTITY WITH NEW MODEL METRICS ON AZURE TABLE STORAGE
    new_entity = {
        'PartitionKey': model_info["name"], 
        'RowKey': model_info["weights"], 
        'Confusion_matrix': model_info['model_metrics']['Confusion_matrix'], 
        'Dataset': model_info['dataset'], 
        'Date': model_info['date'], 
        'F1_class_0': model_info['model_metrics']['F1_score_class_0'], 
        'F1_class_1': model_info['model_metrics']['F1_score_class_1'], 
        'Precision_class_0': model_info['model_metrics']['Precision_score_class_0'], 
        'Precision_class_1': model_info['model_metrics']['Precision_score_class_1'], 
        'Recall_class_0': model_info['model_metrics']['Recall_score_class_0'], 
        'Recall_class_1': model_info['model_metrics']['Recall_score_class_1'], 
        'Total_mAP': model_info['model_metrics']['Total_mAP'], 
        'mAP_class_0': model_info['model_metrics']['mAP_class_0'], 
        'mAP_class_1': model_info['model_metrics']['mAP_class_1'],
        'status': model_info['status'],
        'FPS': model_info['model_metrics']['FPS']
    }

    created_entity = tableClient.create_entity(entity=new_entity)
    print("Created entity: {}".format(created_entity))

    # UPLOAD MODEL WEIGHTS TO AZURE WITH MODEL NAME
    model_file = opt.weights
    model_name = str(model_info['name']) + ".pt"
    blob_client = blob_service_client.get_blob_client(container=container.name, blob=model_name)

    # UPLOAD RESULTS TO CLOUD
    print("Uploading model to cloud")
    with open(model_file, "rb") as data:
        blob_client.upload_blob(data)
    
    print("Model saved!")

# SAVING METRICS RESULTS TO CLOUD AND CHECK IF MODEL HAS BETTER METRICS THAN CURRENT MODEL IN PRODUCTION
def save_results(opt, total_area, total_recall, total_precision, total_f1, confusion, fps):

    model_info = {}
    model_info['name'] = 'model_' + str(int(time.time()))
    model_info['weights'] = opt.weights
    # DATE OF METRICS (dd/mm/YY-H:M:S)
    model_info['date'] = str(datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))
    model_info['dataset'] = str(opt.root)
    model_info['model_metrics'] = {}
    model_info['model_metrics']['Precision_score_class_0'] = str(round(total_precision[0],2))
    model_info['model_metrics']['Recall_score_class_0'] = str(round(total_recall[0],2))
    model_info['model_metrics']['F1_score_class_0'] = str(round(total_f1[0],2))
    model_info['model_metrics']['mAP_class_0'] = str(round(total_area[0],2))
    model_info['model_metrics']['Precision_score_class_1'] = str(round(total_precision[1],2))
    model_info['model_metrics']['Recall_score_class_1'] = str(round(total_recall[1],2))
    model_info['model_metrics']['F1_score_class_1'] = str(round(total_f1[1],2))
    model_info['model_metrics']['mAP_class_1'] = str(round(total_area[1],2))
    model_info['model_metrics']['Total_mAP'] = str(round(mean(total_area),2))
    model_info['model_metrics']['Confusion_matrix'] = str(confusion)
    model_info['model_metrics']['FPS'] = str(round(fps,2))

    # CREATE CONNECTION CLIENTS WITH AZURE
    blob_service_client, container, tableClient = create_azure_connection()

    # CHECK METRICS WITH CURRENT MODEL IN PRODUCTION
    check_model_on_production(opt, blob_service_client, container, tableClient, model_info)

def metrics(opt, fps):

    total_area, total_recall, total_precision, total_f1, confusion = calculate_mAP(opt)
    if opt.check:
        save_results(opt, total_area, total_recall, total_precision, total_f1, confusion, fps)
