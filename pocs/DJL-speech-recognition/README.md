### Build 
```bash
./mvnw clean install 
```
### Result
```
2024-04-23 21:44:30 DEBUG DefaultModelZoo:67 - Scanning models in repo: class ai.djl.repository.SimpleUrlRepository, https://resources.djl.ai/test-models/pytorch/wav2vec2.zip
2024-04-23 21:44:32 DEBUG ModelZoo:111 - Loading model with Criteria:
        Application: UNDEFINED
        Input: class ai.djl.modality.audio.Audio
        Output: class java.lang.String
        Engine: PyTorch
        ModelZoo: ai.djl.localmodelzoo

2024-04-23 21:44:32 DEBUG ModelZoo:115 - Searching model in specified model zoo: ai.djl.localmodelzoo
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-23 21:44:32 DEBUG Engine:165 - Registering EngineProvider: TensorFlow
2024-04-23 21:44:32 DEBUG Engine:95 - Found default engine: PyTorch
2024-04-23 21:44:35 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.localmodelzoo:wav2vec2 UNDEFINED [
        ai.djl.localmodelzoo/wav2vec2/wav2vec2 {}
]
2024-04-23 21:44:35 DEBUG MRL:267 - Preparing artifact: https://resources.djl.ai/test-models/pytorch/wav2vec2.zip, ai.djl.localmodelzoo/wav2vec2/wav2vec2 {}
2024-04-23 21:44:35 DEBUG AbstractRepository:150 - Items to download: 1
2024-04-23 21:44:35 DEBUG SimpleUrlRepository:102 - Downloading artifact: https://resources.djl.ai/test-models/pytorch/wav2vec2.zip ...
2024-04-23 21:44:38 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:44:38 DEBUG LibUtils:410 - Using cache dir: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64
2024-04-23 21:44:38 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libgomp-52f2fd74.so.1
2024-04-23 21:44:38 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libc10.so
2024-04-23 21:44:38 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libtorch_cpu.so
2024-04-23 21:44:39 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/libtorch.so
2024-04-23 21:44:39 DEBUG LibUtils:374 - Loading native library: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/pytorch/2.1.1-cpu-linux-x86_64/0.27.0-libdjl_torch.so
2024-04-23 21:44:39 INFO  PtEngine:67 - PyTorch graph executor optimizer is enabled, this may impact your inference latency and throughput. See: https://docs.djl.ai/docs/development/inference_performance_optimization.html#graph-executor-optimization
2024-04-23 21:44:39 INFO  PtEngine:72 - Number of inter-op threads is 6
2024-04-23 21:44:39 INFO  PtEngine:73 - Number of intra-op threads is 6
2024-04-23 21:44:39 DEBUG JniUtils:1739 - mapLocation: false
2024-04-23 21:44:39 DEBUG JniUtils:1740 - extraFileKeys: []
2024-04-23 21:44:40 INFO  Main:19 - Result: THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT DOOR TO YOU SHALL I CALL ON HIM AS I PASS
```
