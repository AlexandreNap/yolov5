# Install python
FROM public.ecr.aws/lambda/python:3.9

# Install dependencies, exclude dev dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#fix opencv error
RUN pip install opencv-python-headless

# Copy required files
COPY aws_lambda/app .
#RUN pip install -r requirements_lambda.txt
RUN pip install boto3

RUN mkdir -p models
COPY models models
RUN mkdir -p utils
COPY utils utils

COPY detect.py .
COPY export.py .
COPY weights/yolov5l_k26_best.pt model.pt
# COPY authorizer ./

# Set entry point
CMD ["lambda_predict.lambda_handler"]

# debug with docker run --rm -it --entrypoint bash yolo:latest
# build with docker build -t yolo -f aws_lambda/Dockerfile .
# build sam: sam build -t template_no_auth.yaml
