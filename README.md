# sb extension of aws pytorch container for face detection

*(the original readme is
[here](https://github.com/Footage-Firm/sagemaker-pytorch-container/blob/master/)).*


this repo is a fork of the aws pytorch inferent container. use this repo for
local development and to create the container that we user for batch face
detection.

high level, I did the following

1. start with a base that is one of the containers defined here in `docker` and
   used by `aws` for `sagemaker` instances (c.f.
   [aws documentation](https://github.com/aws/deep-learning-containers/blob/master/available_images.md))
1. build a new container that adds into that base our face detection model code
1. iterated on it locally until it was serving
1. pushed that container to ECR
1. launched a sagemaker batch transform job using our ECR container 


## important files

the only files we have added are 3 new files in the `storyblocks` directory

+ `Dockerfile.cpu`: for building our face detection container on a cpu 
+ `Dockerfile.gpu`: same, bug for gpu
+ `inference.py`: the inference code that ultimately defines how the model is
  served

there is also an implicit dependency on the
[DSFD-Pytorch-Inference](https://github.com/hukkelas/DSFD-Pytorch-Inference/)
python code repository. this package is the backbone of our `inference.py` file
and is included via an explicit `git clone` in the `Dockerfile`s


## doing local development

*this was all hacks so please by all means make it better*

first, I built a local version of the pytorch container using the dockerfiles
available in the `docker` directory. 


```shell script
docker build -f Dockerfile.cpu -t sagemaker-pytorch-inference:1_5_0_py3_cpu .
```

this new container became the base of my face detection container, i.e. I used

```dockerfile
FROM sagemaker-pytorch-inference:1_5_0_py3_cpu
```

as the base image in `storyblocks/Dockerfile.{cpu,gpu}` file.

I then built and ran a face detection container via a pycharm run config, which
amounted to the command

```shell script
docker build -f Dockerfile.cpu -t sb-face-detect .
  && docker run \
    -p 8080:8080 -p 8081:8081 \
    --env SAGEMAKER_PROGRAM=inference.py \
    --env AWS_ACCESS_KEY_ID={FILL THIS IN} \
    --env AWS_SECRET_ACCESS_KEY={FILL THIS IN} \
    --env MODEL_NAME=RetinaNetResNet50 \
    --env WITH_LANDMARKS=TRUE \
    --name sb-face-detect \
    --rm \
    sb-face-detect serve
```

where

+ the `AWS_` keys should be supplied by you of course (used for downloading
  files from s3)
+ `MODEL_NAME` should be either `RetinaNetResNet50` or `DSFDDetector`
+ if `WITH_LANDMARKS=TRUE` and `MODEL_NAME=RetinaNetResNet50`, we will return
  face landmarks
    + any other value of `WITH_LANDMARKS` will not attempt to return landmarks
    + `WITH_LANDMARKS=TRUE` and a different model name will result in an error


### local development test commands

I ran the following commands to test that my model was working
 
```shell script
# definitely must work
curl localhost:8080/invocations -H "Content-Type: text/csv" -d "videoblocks-ml/data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"
curl localhost:8080/invocations -H "Content-Type: application/json" -d '{"bucket":"videoblocks-ml","key":"data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg"}'

# interested in knowing if the number of faces changes...
curl localhost:8080/invocations -H "Content-Type: text/csv" -d "videoblocks-ml/data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000131716/000036-1.5015.jpg"
```


## building and deploying to ECR

### building an ECR-compatible image

I replaced the `FROM` line in my dockerfiles with

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04-v1.0
```

for the cpu and

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04-v1.0
```

for the gpu.

note that these are *not* the most current pytorch serving containers per
[aws documentation](https://github.com/aws/deep-learning-containers/blob/master/available_images.md),
but they are the most recent one for which this particular `face_detection`
model is compatible.

following the instructions on that aws documentation page, I pushed my container
to ECR:

```shell script
# ASSUMING AWS CLI V2!!!!!!!!
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04-v1.0
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04-v1.0
```

then `docker build` and `docker run` to your heart's content.


### deploying to ECR

I chose a tag naming convention that shortened the aws version, so instead of
exactly matching `1.5.0-cpu-py36-ubuntu16.04-v1.0`, I went with just
`1.5.0-cpu-py3` and `1.5.0-gpu-py3`. I pushed them to **storyblocks** ecr with

```shell script
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 031780582162.dkr.ecr.us-east-1.amazonaws.com
docker tag sb-face-detect:1.5.0-gpu-py3 031780582162.dkr.ecr.us-east-1.amazonaws.com/sb-facedetect:1.5.0-gpu-py3
docker tag sb-face-detect:1.5.0-cpu-py3 031780582162.dkr.ecr.us-east-1.amazonaws.com/sb-facedetect:1.5.0-cpu-py3
docker push 031780582162.dkr.ecr.us-east-1.amazonaws.com/sb-facedetect:1.5.0-gpu-py3
docker push 031780582162.dkr.ecr.us-east-1.amazonaws.com/sb-facedetect:1.5.0-cpu-py3
```


## deploying a live sagemaker endpoint

first, make sure you have `sagemaker` sdk version 2+. if you are developing code
in this repo, know that this repo uses v1, so you may need to switch
environments.

```python
from sagemaker.model import Model
from sagemaker.pytorch.model import PyTorchPredictor

env = {'SAGEMAKER_PROGRAM': 'inference.py',
       'MODEL_NAME': 'RetinaNetResNet50',
       'WITH_LANDMARKS': 'TRUE', }
# 'CONFIDENCE_THRESHOLD': 0.5
# 'NMS_IOU_THRESHOLD': 0.3

model = Model(image_uri='031780582162.dkr.ecr.us-east-1.amazonaws.com/sb-facedetect:1.5.0-gpu-py3',
              model_data=None,
              role='sagemaker-ds',
              predictor_cls=PyTorchPredictor,
              env=env)

predictor = model.deploy(initial_instance_count=1,
                         instance_type='ml.p3.2xlarge')

# hard to predictor.predict here for reasons

import boto3
import json

runtime= boto3.client('runtime.sagemaker')
bucket = 'videoblocks-ml'
key = 'data/object-detection-research/videoblocks/dev/sampled-items/jpg/fps-method-01/000023419/000216-9.0090.jpg'

# csv test
result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                 Body=f"{bucket}/{key}",
                                 ContentType='text/csv')
print(result["Body"].read())

# json test
result = runtime.invoke_endpoint(EndpointName=model.endpoint_name,
                                 Body=json.dumps({'bucket': bucket, 'key': key}),
                                 ContentType='application/json')
print(result["Body"].read())


# bring 'er down cap'n
predictor.delete_endpoint()
model.delete_model()
```
