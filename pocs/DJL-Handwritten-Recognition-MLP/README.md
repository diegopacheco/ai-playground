### Build 
```bash
./mvnw clean install 
```
### Result
```
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-23 21:28:03 DEBUG Engine:165 - Registering EngineProvider: TensorFlow
2024-04-23 21:28:03 DEBUG Engine:95 - Found default engine: PyTorch
2024-04-23 21:28:06 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:28:06 DEBUG LibUtils:410 - Using cache dir: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64
2024-04-23 21:28:06 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libgomp-52f2fd74.so.1
2024-04-23 21:28:06 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libc10.so
2024-04-23 21:28:06 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libtorch_cpu.so
2024-04-23 21:28:06 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libtorch.so
2024-04-23 21:28:06 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/0.27.0-libdjl_torch.so
2024-04-23 21:28:06 INFO  PtEngine:67 - PyTorch graph executor optimizer is enabled, this may impact your inference latency and throughput. See: https://docs.djl.ai/docs/development/inference_performance_optimization.html#graph-executor-optimization
2024-04-23 21:28:06 INFO  PtEngine:72 - Number of inter-op threads is 6
2024-04-23 21:28:06 INFO  PtEngine:73 - Number of intra-op threads is 6
2024-04-23 21:28:07 DEBUG MRL:267 - Preparing artifact: BasicDataset, ai.djl.basicdataset/mnist/1.0/null {}
2024-04-23 21:28:07 DEBUG AbstractRepository:126 - Files have been downloaded already: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/cache/repo/dataset/cv/ai/djl/basicdataset/mnist/1.0
2024-04-23 21:28:07 DEBUG MRL:267 - Preparing artifact: BasicDataset, ai.djl.basicdataset/mnist/1.0/null {}
2024-04-23 21:28:07 DEBUG AbstractRepository:126 - Files have been downloaded already: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/cache/repo/dataset/cv/ai/djl/basicdataset/mnist/1.0
2024-04-23 21:28:07 INFO  LoggingTrainingListener:160 - Training on: cpu().
2024-04-23 21:28:07 INFO  LoggingTrainingListener:167 - Load PyTorch Engine Version 2.1.1 in 0.030 ms.
Training:    100% |████████████████████████████████████████| Accuracy: 0.92, SoftmaxCrossEntropyLoss: 0.27, speed: 16763.18 items/sec
Validating:  100% |████████████████████████████████████████|
2024-04-23 21:28:10 INFO  LoggingTrainingListener:67 - Epoch 1 finished.
2024-04-23 21:28:10 INFO  LoggingTrainingListener:78 - Train: Accuracy: 0.92, SoftmaxCrossEntropyLoss: 0.27
2024-04-23 21:28:10 INFO  LoggingTrainingListener:91 - Validate: Accuracy: 0.96, SoftmaxCrossEntropyLoss: 0.14
Training:    100% |████████████████████████████████████████| Accuracy: 0.97, SoftmaxCrossEntropyLoss: 0.11, speed: 16328.51 items/sec
Validating:  100% |████████████████████████████████████████|
2024-04-23 21:28:12 INFO  LoggingTrainingListener:67 - Epoch 2 finished.
2024-04-23 21:28:12 INFO  LoggingTrainingListener:78 - Train: Accuracy: 0.97, SoftmaxCrossEntropyLoss: 0.11
2024-04-23 21:28:12 INFO  LoggingTrainingListener:91 - Validate: Accuracy: 0.97, SoftmaxCrossEntropyLoss: 0.10
Training:    100% |████████████████████████████████████████| Accuracy: 0.98, SoftmaxCrossEntropyLoss: 0.08, speed: 7955.83 items/secc
Validating:  100% |████████████████████████████████████████|
2024-04-23 21:28:15 INFO  LoggingTrainingListener:67 - Epoch 3 finished.
2024-04-23 21:28:15 INFO  LoggingTrainingListener:78 - Train: Accuracy: 0.98, SoftmaxCrossEntropyLoss: 0.08
2024-04-23 21:28:15 INFO  LoggingTrainingListener:91 - Validate: Accuracy: 0.97, SoftmaxCrossEntropyLoss: 0.08
Training:    100% |████████████████████████████████████████| Accuracy: 0.98, SoftmaxCrossEntropyLoss: 0.06, speed: 12795.06 items/sec
Validating:  100% |████████████████████████████████████████|
2024-04-23 21:28:17 INFO  LoggingTrainingListener:67 - Epoch 4 finished.
2024-04-23 21:28:17 INFO  LoggingTrainingListener:78 - Train: Accuracy: 0.98, SoftmaxCrossEntropyLoss: 0.06
2024-04-23 21:28:17 INFO  LoggingTrainingListener:91 - Validate: Accuracy: 0.97, SoftmaxCrossEntropyLoss: 0.09
Training:    100% |████████████████████████████████████████| Accuracy: 0.99, SoftmaxCrossEntropyLoss: 0.05, speed: 14404.61 items/sec
Validating:  100% |████████████████████████████████████████|
2024-04-23 21:28:19 INFO  LoggingTrainingListener:67 - Epoch 5 finished.
2024-04-23 21:28:19 INFO  LoggingTrainingListener:78 - Train: Accuracy: 0.99, SoftmaxCrossEntropyLoss: 0.05
2024-04-23 21:28:19 INFO  LoggingTrainingListener:91 - Validate: Accuracy: 0.98, SoftmaxCrossEntropyLoss: 0.08
2024-04-23 21:28:19 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:28:19 INFO  LoggingTrainingListener:187 - train P50: 2.398 ms, P90: 3.318 ms
2024-04-23 21:28:19 INFO  LoggingTrainingListener:193 - forward P50: 0.271 ms, P90: 0.364 ms
2024-04-23 21:28:19 INFO  LoggingTrainingListener:199 - training-metrics P50: 0.005 ms, P90: 0.008 ms
2024-04-23 21:28:19 INFO  LoggingTrainingListener:205 - backward P50: 0.613 ms, P90: 0.866 ms
2024-04-23 21:28:19 INFO  LoggingTrainingListener:211 - step P50: 0.832 ms, P90: 1.171 ms
2024-04-23 21:28:19 INFO  LoggingTrainingListener:217 - epoch P50: 2.448 s, P90: 2.896 s
Model training complete. Model saved!
ai.djl.metric.Metrics@33abac0c
F.Y.I the number is 5 - lets see is AI knows...
2024-04-23 21:28:19 DEBUG BaseModel:341 - Try to load model from /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/git/diegopacheco/ai-playground/pocs/DJL-Handwritten-Recognition-MLP/mlp_model/mlp-0005.params
2024-04-23 21:28:19 DEBUG BaseModel:360 - Loading saved model: mlp parameter
2024-04-23 21:28:19 DEBUG BaseModel:380 - DJL model loaded successfully
[
        {"class": "5", "probability": 0.71182}
        {"class": "4", "probability": 0.28570}
        {"class": "8", "probability": 0.23505}
        {"class": "2", "probability": 0.11023}
        {"class": "7", "probability": 0.09434}
]
```