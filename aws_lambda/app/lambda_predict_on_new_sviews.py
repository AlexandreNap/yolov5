import io
import logging
import cloud_utils
from datetime import datetime, timezone
import torch
from PIL import Image
from PIL import ImageFile
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_image_from_s3(bucket, key):
    """Load image file from s3.
    """
    object = bucket.Object(key)
    try:
        img_data = object.get().get('Body').read()
        img = Image.open(io.BytesIO(img_data))
        img.verify()  # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
        return None
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((1920, 1080))
    return img


def get_sviews(col, bucket):
    date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    query = {"n_surfers_yolo": None, "date": date}
    db_list = []
    for x in col.find(query):
        db_list.append({"id": x["_id"],
                        "path": x["s3_path"]})
    db_list = db_list[np.max((0, -7)):]

    input_list = []
    for x in db_list:
        path = x["path"]
        img = load_image_from_s3(bucket, path)
        input_list.append({"id": x["id"],
                           "img": img,
                           "path": path})
    return input_list


def lambda_handler(event, context):
    client = cloud_utils.mongo_client()
    db = client["surf"]
    col = db["sviews"]
    bucket = cloud_utils.bucket_resource()

    input_list = get_sviews(col, bucket)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model = torch.hub.load('./', 'custom',
                           path="model.pt", source='local')
    model.conf = 0.33  # NMS confidence threshold

    for input in input_list:
        img = input["img"]
        if img is not None:
            results = model([img], size=1920)
            print(len(results.pandas().xyxy[0]))
            print(input["path"])
            col.update_one({"_id": input["id"]}, {"$set": {"n_surfers_yolo5":  len(results.pandas().xyxy[0]), "model_name": "yolov5l_k32_best"}})

    return {
        'statusCode': 200,
        'headers': {
            'Content-type': 'application/json'
        },
        'body': -1
    }
