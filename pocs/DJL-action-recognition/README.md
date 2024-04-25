### Build 
```bash
./mvnw clean install 
```
### Input
<img src="src/main/resources/action_discus_throw.png" />

### Result Summary
```
[
    {"class": "ThrowDiscus", "probability": 0.99868}
    {"class": "Hammering", "probability": 0.00131}
    {"class": "JavelinThrow", "probability": 7.4e-09}
    {"class": "VolleyballSpiking", "probability": 1.8e-10}
    {"class": "LongJump", "probability": 5.8e-11}
]
```

### Result Full
```
2024-04-24 20:20:34 DEBUG ModelZoo:111 - Loading model with Criteria:
        Application: CV.ACTION_RECOGNITION
        Input: interface ai.djl.modality.cv.Image
        Output: class ai.djl.modality.Classifications
        Engine: MXNet
        Filter: {"backbone":"inceptionv3","dataset":"ucf101"}
        No translator supplied

2024-04-24 20:20:34 DEBUG ModelZoo:138 - Ignore ModelZoo ai.djl.pytorch by engine: MXNet
2024-04-24 20:20:34 DEBUG ModelZoo:138 - Ignore ModelZoo ai.djl.onnxruntime by engine: MXNet
2024-04-24 20:20:34 DEBUG ModelZoo:138 - Ignore ModelZoo ai.djl.tensorflow by engine: MXNet
2024-04-24 20:20:34 DEBUG ModelZoo:138 - Ignore ModelZoo ai.djl.huggingface.pytorch by engine: MXNet
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: TensorFlow
2024-04-24 20:20:34 DEBUG Engine:165 - Registering EngineProvider: MXNet
2024-04-24 20:20:34 DEBUG Engine:95 - Found default engine: MXNet
2024-04-24 20:20:37 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.zoo:mlp CV.IMAGE_CLASSIFICATION [
        ai.djl.zoo/mlp/0.0.3/mlp {"dataset":"mnist"}
]
2024-04-24 20:20:37 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.zoo:mlp
2024-04-24 20:20:37 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.zoo:resnet CV.IMAGE_CLASSIFICATION [
        ai.djl.zoo/resnet/0.0.2/resnetv1 {"layers":"50","flavor":"v1","dataset":"cifar10"}
]
2024-04-24 20:20:37 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.zoo:resnet
2024-04-24 20:20:37 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.zoo:ssd CV.OBJECT_DETECTION [
        ai.djl.zoo/ssd/0.0.2/ssd {"flavor":"tiny","dataset":"pikachu"}
]
2024-04-24 20:20:37 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.zoo:ssd
2024-04-24 20:20:37 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:yolo CV.OBJECT_DETECTION [
        ai.djl.mxnet/yolo/0.0.1/yolo {"dataset":"voc","version":"3","backbone":"darknet53","imageSize":"320"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_darknet_voc_416 {"dataset":"voc","version":"3","backbone":"darknet53","imageSize":"416"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_mobilenet_voc_320 {"dataset":"voc","version":"3","backbone":"mobilenet1.0","imageSize":"320"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_mobilenet_voc_41 {"dataset":"voc","version":"3","backbone":"mobilenet1.0","imageSize":"416"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_darknet_coco_320 {"dataset":"coco","version":"3","backbone":"darknet53","imageSize":"320"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_darknet_coco_416 {"dataset":"coco","version":"3","backbone":"darknet53","imageSize":"416"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_darknet_coco_608 {"dataset":"coco","version":"3","backbone":"darknet53","imageSize":"608"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_mobilenet_coco_320 {"dataset":"coco","version":"3","backbone":"mobilenet1.0","imageSize":"320"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_mobilenet_coco_416 {"dataset":"coco","version":"3","backbone":"mobilenet1.0","imageSize":"416"}
        ai.djl.mxnet/yolo/0.0.1/yolo3_mobilenet_coco_608 {"dataset":"coco","version":"3","backbone":"mobilenet1.0","imageSize":"608"}
]
2024-04-24 20:20:37 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:yolo
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:ssd CV.OBJECT_DETECTION [
        ai.djl.mxnet/ssd/0.0.1/ssd_512_resnet50_v1_voc {"size":"512","backbone":"resnet50","flavor":"v1","dataset":"voc"}
        ai.djl.mxnet/ssd/0.0.1/ssd_512_vgg16_atrous_coco {"size":"512","backbone":"vgg16","flavor":"atrous","dataset":"coco"}
        ai.djl.mxnet/ssd/0.0.1/ssd_512_mobilenet1.0_voc {"size":"512","backbone":"mobilenet1.0","dataset":"voc"}
        ai.djl.mxnet/ssd/0.0.1/ssd_300_vgg16_atrous_voc {"size":"300","backbone":"vgg16","flavor":"atrous","dataset":"voc"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:ssd
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:googlenet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/googlenet/0.0.1/googlenet {"dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:googlenet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:glove NLP.WORD_EMBEDDING [
        ai.djl.mxnet/glove/0.0.2/glove {"dimensions":"50"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:glove
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:darknet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/darknet/0.0.1/darknet53 {"layers":"53","flavor":"v3","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:darknet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:squeezenet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/squeezenet/0.0.1/squeezenet1.0 {"flavor":"1.0","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:squeezenet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:bertqa NLP.QUESTION_ANSWER [
        ai.djl.mxnet/bertqa/0.0.1/static_bert_qa {"backbone":"bert","dataset":"book_corpus_wiki_en_uncased"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:bertqa
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:inceptionv3 CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/inceptionv3/0.0.1/inceptionv3 {"dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:inceptionv3
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:mask_rcnn CV.INSTANCE_SEGMENTATION [
        ai.djl.mxnet/mask_rcnn/0.0.1/mask_rcnn_resnet18_v1b_coco {"backbone":"resnet18","flavor":"v1b","dataset":"coco"}
        ai.djl.mxnet/mask_rcnn/0.0.1/mask_rcnn_resnet101_v1d_coco {"backbone":"resnet101","flavor":"v1d","dataset":"coco"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:mask_rcnn
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:mobilenet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/mobilenet/0.0.1/mobilenet0.25 {"flavor":"v1","multiplier":"0.25","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenet0.5 {"flavor":"v1","multiplier":"0.5","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenet0.75 {"flavor":"v1","multiplier":"0.75","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenet1.0 {"flavor":"v1","multiplier":"1.0","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv2_0.25 {"flavor":"v2","multiplier":"0.25","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv2_0.5 {"flavor":"v2","multiplier":"0.5","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv2_0.75 {"flavor":"v2","multiplier":"0.75","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv2_1.0 {"flavor":"v2","multiplier":"1.0","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv3_small {"flavor":"v3_small","multiplier":"1.0","dataset":"imagenet"}
        ai.djl.mxnet/mobilenet/0.0.1/mobilenetv3_large {"flavor":"v3_large","multiplier":"1.0","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:mobilenet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:senet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/senet/0.0.1/senet_154 {"layers":"154","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:senet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:alexnet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/alexnet/0.0.1/alexnet {"dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:alexnet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:mlp CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/mlp/0.0.1/mlp {"dataset":"mnist"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:mlp
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:resnet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/resnet/0.0.1/resnet18_v1 {"layers":"18","flavor":"v1","dataset":"imagenet"}
        ai.djl.mxnet/resnet/0.0.1/resnet50_v2 {"layers":"50","flavor":"v2","dataset":"imagenet"}
        ai.djl.mxnet/resnet/0.0.1/resnet101_v1 {"layers":"101","dataset":"imagenet"}
        ai.djl.mxnet/resnet/0.0.1/resnet152_v1d {"layers":"152","flavor":"v1d","dataset":"imagenet"}
        ai.djl.mxnet/resnet/0.0.1/resnet50_v1_cifar10 {"layers":"50","flavor":"v1","dataset":"cifar10"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:resnet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:simple_pose CV.POSE_ESTIMATION [
        ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b {"backbone":"resnet18","flavor":"v1b","dataset":"imagenet"}
        ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet50_v1b {"backbone":"resnet50","flavor":"v1b","dataset":"imagenet"}
        ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet101_v1d {"backbone":"resnet101","flavor":"v1d","dataset":"imagenet"}
        ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet152_v1b {"backbone":"resnet152","flavor":"v1b","dataset":"imagenet"}
        ai.djl.mxnet/simple_pose/0.0.1/simple_pose_resnet152_v1d {"backbone":"resnet152","flavor":"v1d","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:simple_pose
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:se_resnext CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/se_resnext/0.0.1/se_resnext101_32x4d {"layers":"101","flavor":"32x4d","dataset":"imagenet"}
        ai.djl.mxnet/se_resnext/0.0.1/se_resnext101_64x4d {"layers":"101","flavor":"64x4d","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:se_resnext
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:vgg CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/vgg/0.0.1/vgg11 {"layers":"11","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg13 {"layers":"13","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg16 {"layers":"16","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg19 {"layers":"19","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg11_bn {"flavor":"batch_norm","layers":"11","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg13_bn {"flavor":"batch_norm","layers":"13","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg16_bn {"flavor":"batch_norm","layers":"16","dataset":"imagenet"}
        ai.djl.mxnet/vgg/0.0.1/vgg19_bn {"flavor":"batch_norm","layers":"19","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:vgg
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:densenet CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/densenet/0.0.1/densenet121 {"layers":"121","dataset":"imagenet"}
        ai.djl.mxnet/densenet/0.0.1/densenet161 {"layers":"161","dataset":"imagenet"}
        ai.djl.mxnet/densenet/0.0.1/densenet169 {"layers":"169","dataset":"imagenet"}
        ai.djl.mxnet/densenet/0.0.1/densenet201 {"layers":"201","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:densenet
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:xception CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/xception/0.0.1/xception {"flavor":"65","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:xception
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:deepar TIMESERIES.FORECASTING [
        ai.djl.mxnet/deepar/0.0.1/airpassengers {"dataset":"airpassengers"}
        ai.djl.mxnet/deepar/0.0.1/m5forecast {"dataset":"m5forecast"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:deepar
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:resnest CV.IMAGE_CLASSIFICATION [
        ai.djl.mxnet/resnest/0.0.1/resnest14 {"layers":"14","dataset":"imagenet"}
        ai.djl.mxnet/resnest/0.0.1/resnest26 {"layers":"26","dataset":"imagenet"}
        ai.djl.mxnet/resnest/0.0.1/resnest50 {"layers":"50","dataset":"imagenet"}
        ai.djl.mxnet/resnest/0.0.1/resnest101 {"layers":"101","dataset":"imagenet"}
        ai.djl.mxnet/resnest/0.0.1/resnest200 {"layers":"200","dataset":"imagenet"}
        ai.djl.mxnet/resnest/0.0.1/resnest269 {"layers":"269","dataset":"imagenet"}
]
2024-04-24 20:20:38 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.mxnet:resnest
2024-04-24 20:20:38 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.mxnet:action_recognition CV.ACTION_RECOGNITION [
        ai.djl.mxnet/action_recognition/0.0.1/vgg16_ucf101 {"backbone":"vgg16","dataset":"ucf101"}
        ai.djl.mxnet/action_recognition/0.0.1/inceptionv3_ucf101 {"backbone":"inceptionv3","dataset":"ucf101"}
]
2024-04-24 20:20:38 DEBUG MRL:267 - Preparing artifact: MXNet, ai.djl.mxnet/action_recognition/0.0.1/inceptionv3_ucf101 {"backbone":"inceptionv3","dataset":"ucf101"}
2024-04-24 20:20:38 DEBUG AbstractRepository:150 - Items to download: 3
2024-04-24 20:20:38 DEBUG AbstractRepository:189 - Downloading artifact: https://mlrepo.djl.ai/model/cv/action_recognition/ai/djl/mxnet/action_recognition/classes.txt ...
Downloading:   0% |█                                       |2024-04-24 20:20:38 DEBUG AbstractRepository:189 - Downloading artifact: https://mlrepo.djl.ai/model/cv/action_recognition/ai/djl/mxnet/action_recognition/0.0.1/inceptionv3_ucf101-symbol.json ...
Downloading:   0% |█                                       |2024-04-24 20:20:38 DEBUG AbstractRepository:189 - Downloading artifact: https://mlrepo.djl.ai/model/cv/action_recognition/ai/djl/mxnet/action_recognition/0.0.1/inceptionv3_ucf101-0000.params.gz ...
Downloading: 100% |████████████████████████████████████████|
Loading:     100% |████████████████████████████████████████|
2024-04-24 20:20:40 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-24 20:20:41 DEBUG LibUtils:256 - Using cache dir: /home/diego/.djl.ai/mxnet/1.9.1-mkl-linux-x86_64
2024-04-24 20:20:41 INFO  LibUtils:274 - Downloading libgfortran.so.3 ...
2024-04-24 20:20:41 INFO  LibUtils:274 - Downloading libgomp.so.1 ...
2024-04-24 20:20:41 INFO  LibUtils:274 - Downloading libquadmath.so.0 ...
2024-04-24 20:20:41 INFO  LibUtils:274 - Downloading libopenblas.so.0 ...
2024-04-24 20:20:41 INFO  LibUtils:274 - Downloading libmxnet.so ...
2024-04-24 20:20:42 DEBUG LibUtils:68 - Loading mxnet library from: /home/diego/.djl.ai/mxnet/1.9.1-mkl-linux-x86_64/libmxnet.so
[20:20:42] ../src/nnvm/legacy_json_util.cc:209: Loading symbol saved by previous version v1.5.0. Attempting to upgrade...
[20:20:42] ../src/nnvm/legacy_json_util.cc:217: Symbol successfully upgraded!
2024-04-24 20:20:42 DEBUG BaseModel:341 - Try to load model from /home/diego/.djl.ai/cache/repo/model/cv/action_recognition/ai/djl/mxnet/action_recognition/inceptionv3/ucf101/0.0.1/inceptionv3_ucf101-0000.params
2024-04-24 20:20:42 DEBUG MxModel:212 - DJL formatted model not found, try to find MXNet model
2024-04-24 20:20:42 DEBUG MxModel:235 - MXNet Model /home/diego/.djl.ai/cache/repo/model/cv/action_recognition/ai/djl/mxnet/action_recognition/inceptionv3/ucf101/0.0.1/inceptionv3_ucf101-0000.params (float32) loaded successfully.
2024-04-24 20:20:43 INFO  Main:24 - [
        {"class": "ThrowDiscus", "probability": 0.99868}
        {"class": "Hammering", "probability": 0.00131}
        {"class": "JavelinThrow", "probability": 7.4e-09}
        {"class": "VolleyballSpiking", "probability": 1.8e-10}
        {"class": "LongJump", "probability": 5.8e-11}
]
```