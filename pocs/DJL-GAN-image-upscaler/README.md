### Build 
```bash
./mvnw clean install 
```
### Input (128x128)
<img src="src/main/resources/fox.png" >

### Output (515x512)
<img src="build/output/super-res/image0.png" >

### Result
```
2024-04-23 21:52:48 DEBUG DefaultModelZoo:67 - Scanning models in repo: class ai.djl.repository.SimpleUrlRepository, https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz
2024-04-23 21:52:49 DEBUG ModelZoo:111 - Loading model with Criteria:
        Application: CV.IMAGE_ENHANCEMENT
        Input: interface ai.djl.modality.cv.Image
        Output: interface ai.djl.modality.cv.Image
        Engine: TensorFlow
        ModelZoo: ai.djl.localmodelzoo
        Options: {"SignatureDefKey":"serving_default","Tags":"serve"}

2024-04-23 21:52:49 DEBUG ModelZoo:115 - Searching model in specified model zoo: ai.djl.localmodelzoo
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-23 21:52:49 DEBUG Engine:165 - Registering EngineProvider: TensorFlow
2024-04-23 21:52:49 DEBUG Engine:95 - Found default engine: PyTorch
2024-04-23 21:52:52 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.localmodelzoo:1 UNDEFINED [
        ai.djl.localmodelzoo/1/1 {}
]
2024-04-23 21:52:52 DEBUG MRL:267 - Preparing artifact: https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz, ai.djl.localmodelzoo/1/1 {}
2024-04-23 21:52:52 DEBUG AbstractRepository:150 - Items to download: 1
2024-04-23 21:52:52 DEBUG SimpleUrlRepository:102 - Downloading artifact: https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz ...
Downloading: 100% |████████████████████████████████████████|
Loading:     100% |████████████████████████████████████████|
2024-04-23 21:52:53 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:52:53 DEBUG LibUtils:162 - Using cache dir: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/tensorflow
2024-04-23 21:52:54 INFO  LibUtils:224 - Downloading https://publish.djl.ai/tensorflow-2.10.1/linux/cpu/THIRD_PARTY_TF_JNI_LICENSES.gz ...
2024-04-23 21:52:54 INFO  LibUtils:224 - Downloading https://publish.djl.ai/tensorflow-2.10.1/linux/cpu/LICENSE.gz ...
2024-04-23 21:52:54 INFO  LibUtils:224 - Downloading https://publish.djl.ai/tensorflow-2.10.1/linux/cpu/libjnitensorflow.so.gz ...
2024-04-23 21:52:55 INFO  LibUtils:224 - Downloading https://publish.djl.ai/tensorflow-2.10.1/linux/cpu/libtensorflow_framework.so.2.gz ...
2024-04-23 21:52:55 INFO  LibUtils:224 - Downloading https://publish.djl.ai/tensorflow-2.10.1/linux/cpu/libtensorflow_cc.so.2.gz ...
2024-04-23 21:52:58 DEBUG LibUtils:50 - Loading TensorFlow library from: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/tensorflow/2.10.1-cpu-linux-x86_64/libjnitensorflow.so
2024-04-23 21:52:58 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:52:59.055149: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-23 21:52:59 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-23 21:52:59.104747: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/cache/repo/model/undefined/ai/djl/localmodelzoo/2bf7ee9a0db7fde2dee7c9ce10f5e441a83b443c
2024-04-23 21:52:59.185194: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:89] Reading meta graph with tags { serve }
2024-04-23 21:52:59.185231: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:130] Reading SavedModel debug info (if present) from: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/cache/repo/model/undefined/ai/djl/localmodelzoo/2bf7ee9a0db7fde2dee7c9ce10f5e441a83b443c
2024-04-23 21:52:59.288393: I external/org_tensorflow/tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
2024-04-23 21:52:59.305583: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:229] Restoring SavedModel bundle.
2024-04-23 21:52:59.969720: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:213] Running initialization op on SavedModel bundle at path: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/DJL_CACHE_DIR/cache/repo/model/undefined/ai/djl/localmodelzoo/2bf7ee9a0db7fde2dee7c9ce10f5e441a83b443c
2024-04-23 21:53:00.172352: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:305] SavedModel load for tags { serve }; Status: success: OK. Took 1067607 microseconds.
2024-04-23 21:53:03 INFO  Main:41 - Using TensorFlow Engine. 1 images generated.
2024-04-23 21:53:03 INFO  Main:52 - Generated images have been saved in: build/output/super-res
```