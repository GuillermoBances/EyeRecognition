import os
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, TableClient
from zipfile import ZipFile

def download_from_azure(directory):

    # FUNCTION FOR DOWNLOADING YOLO MODEL AND WEIGHTS FROM AZURE BLOB STORAGE WITH CONNECTION STRING
    connect_str = "DefaultEndpointsProtocol=https;AccountName=d2mcapstones00;AccountKey=cHCeAzdLoZKhZi6d/wIzeDJMQJeWF3OsnbcuDgZtskgnsmEXYezGGnSBGGWd+J9hsyVIfDddjtXP+AStPlvNjA==;EndpointSuffix=core.windows.net"
    
    # CHECK ALL CONTAINERS IN AZURE BLOB STORAGE AND SELECT DROWSINESS CONTAINER
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    test_containers = blob_service_client.list_containers()
    for element in test_containers:
        if "drowsiness" in element.name:
            container = element
    container_client = blob_service_client.get_container_client(container) 

    # CHECK ALL TABLES IN AZURE TABLE STORAGE AND SELECT TABLE FOR CHECKING CURRENT MODEL IN PRODUCTION 
    table_service_client = TableServiceClient.from_connection_string(connect_str)
    list_tables = table_service_client.list_tables()
    for element in list_tables:
        if "drowsinessModel" in element.name:
            table = element

    tableClient = TableClient.from_connection_string(connect_str, table_name=table.name)

    # LIST ALL ENTITIES IN TABLE AND CHECK FOR CURRENT MODEL IN PRODUCTION
    entities = list(tableClient.list_entities())
    for entity in entities:
        status = entity["status"]
        if status == "in_production":
            current_model = entity
            model_name = current_model["PartitionKey"]
            break

    # LIST BLOBS IN CONTAINER AND GET YOLO MODEL AND WEIGHTS
    blob_list = container_client.list_blobs()
    for element in blob_list:
        if model_name in element.name:
            weights = element
        elif "ultralytics_yolov5_master" in element.name:
            model = element
    
    # DOWNLOAD YOLO WEIGHTS IN LOCAL
    download_weight_path = os.path.join(directory, weights.name)
    print("Downloading YOLO weights in " + download_weight_path)
    with open(download_weight_path, "wb") as download_file:
        download_file.write(container_client.download_blob(weights.name).readall())

    # DOWNLOAD YOLO MODEL (ZIP) IN LOCAL
    download_model_path = os.path.join(directory, model.name)
    print("Downloading YOLO model in " + download_model_path)
    with open(download_model_path, "wb") as download_file:
        download_file.write(container_client.download_blob(model.name).readall())

    # UNZIP YOLO MODEL
    with ZipFile(download_model_path, 'r') as zipObj:
        zipObj.extractall(directory)

    return weights.name