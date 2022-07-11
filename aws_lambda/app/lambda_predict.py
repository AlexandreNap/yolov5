import io
import json
import boto3
import numpy as np
import logging
import base64
import os
from PIL import Image
from detect import run
from shutil import rmtree

# Define logger class
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
# Helper function to download object from S3 Bucket
def DownloadFromS3(bucket:str, key:str):
    s3 = boto3.client('s3')
    with BytesIO() as f:
        s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)
        f.seek(0)
        test_features = joblib.load(f)
    return test_features

# Load model into memory
logger.info('Loading model from file...')
knnclf = joblib.load('knnclf.joblib')
logger.info('Model Loaded from file...')
"""

"""
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
)"""


def lambda_handler(event, context):
    # Read JSON data packet
    data = json.loads(event['body'])
    images = data['images']

    if not isinstance(images, list):
        logger.info(f'Input should be a list of images, Converting input as [input]')
        images = [images]

    logger.info(f'images folder creating ...')
    if os.path.exists("/tmp/images"):
        rmtree("/tmp/images")
    os.mkdir("/tmp/images")
    logger.info(f'images folder created')

    logger.info(f'Performing images loading and caching...')
    file_names = ["image_" + str(i) for i in range(len(images))]
    for image, name in zip(images, file_names):
        # convert it into bytes
        img_bytes = base64.b64decode(image.encode('utf-8'))

        # convert bytes data to PIL Image object
        img = Image.open(io.BytesIO(img_bytes))
        # PIL image object to numpy array
        img = img.resize((1920, 1080), Image.ANTIALIAS)
        img.save(f"/tmp/images/{name}.jpg")
    logger.info(f'Images loaded!')

    logger.info(f'Performing predictions...')

    if os.path.exists('/tmp/runs/detect/exp/'):
        rmtree('/tmp/runs/detect/exp/')
    run(weights="model.pt", source="/tmp/images/", data=None,
        imgsz=(1080, 1920), project="/tmp/runs/detect",  save_txt=True, nosave=True)
    logger.info(f'Predictions performed!')

    data = []
    for file_name in file_names:
        path = '/tmp/runs/detect/exp/labels/' + file_name + '.txt'
        if os.path.exists(path):
            with open(path, 'r') as file:
                data.append({"pred": file.read().split("\n")[:-1]})
        else:
            data.append({"pred": []})

    response = json.dumps(data)

    return {
        'statusCode': 200,
        'headers': {
            'Content-type': 'application/json'
        },
        'body': response
    }
