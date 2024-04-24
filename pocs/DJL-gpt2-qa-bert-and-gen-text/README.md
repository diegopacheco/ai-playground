### About

This POC uses DLJ from AWS.

* Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning.
* DLJ supports:
  * PyTorch TorchScript model
  * TensorFlow SavedModel bundle
  * Apache MXNet model
  * ONNX model
  * TensorRT model
  * Python script model
  * PaddlePaddle model
  * TFLite model
  * XGBoost model
  * LightGBM model
  * Sentencepiece model
  * fastText/BlazingText model
* 

### Build 
```bash
./mvnw clean install 
```
### Result 

##### Generate Text
GPT2GenerateText
```
2024-04-22 23:44:39 DEBUG DefaultModelZoo:67 - Scanning models in repo: class ai.djl.repository.SimpleUrlRepository, https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip
2024-04-22 23:44:39 DEBUG ModelZoo:111 - Loading model with Criteria:
	Application: UNDEFINED
	Input: class ai.djl.ndarray.NDList
	Output: class ai.djl.modality.nlp.generate.CausalLMOutput
	Engine: PyTorch
	ModelZoo: ai.djl.localmodelzoo

2024-04-22 23:44:39 DEBUG ModelZoo:115 - Searching model in specified model zoo: ai.djl.localmodelzoo
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-22 23:44:39 DEBUG Engine:165 - Registering EngineProvider: TensorFlow
2024-04-22 23:44:39 DEBUG Engine:95 - Found default engine: PyTorch
2024-04-22 23:44:43 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.localmodelzoo:gpt2_pt UNDEFINED [
	ai.djl.localmodelzoo/gpt2_pt/gpt2_pt {}
]
2024-04-22 23:44:43 DEBUG MRL:267 - Preparing artifact: https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip, ai.djl.localmodelzoo/gpt2_pt/gpt2_pt {}
2024-04-22 23:44:43 DEBUG AbstractRepository:126 - Files have been downloaded already: /home/diego/.djl.ai/cache/repo/model/undefined/ai/djl/localmodelzoo/5db362b8f477d219031fab733962dcf4407f55dc
2024-04-22 23:44:43 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:44:43 DEBUG LibUtils:410 - Using cache dir: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64
2024-04-22 23:44:43 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libgomp-52f2fd74.so.1
2024-04-22 23:44:43 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libc10.so
2024-04-22 23:44:43 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libtorch_cpu.so
2024-04-22 23:44:43 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libtorch.so
2024-04-22 23:44:43 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/0.27.0-libdjl_torch.so
2024-04-22 23:44:43 INFO  PtEngine:67 - PyTorch graph executor optimizer is enabled, this may impact your inference latency and throughput. See: https://docs.djl.ai/docs/development/inference_performance_optimization.html#graph-executor-optimization
2024-04-22 23:44:43 INFO  PtEngine:72 - Number of inter-op threads is 6
2024-04-22 23:44:43 INFO  PtEngine:73 - Number of intra-op threads is 6
2024-04-22 23:44:43 DEBUG JniUtils:1739 - mapLocation: false
2024-04-22 23:44:43 DEBUG JniUtils:1740 - extraFileKeys: []
2024-04-22 23:44:44 INFO  DeferredTranslatorFactory:68 - Using TranslatorFactory: ai.djl.pytorch.zoo.nlp.textgeneration.PtGptTranslatorFactory
2024-04-22 23:44:44 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:44:44 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:44:44 INFO  Platform:82 - Found matching platform from: jar:file:/home/diego/.m2/repository/ai/djl/huggingface/tokenizers/0.27.0/tokenizers-0.27.0.jar!/native/lib/tokenizers.properties
2024-04-22 23:44:44 DEBUG LibUtils:88 - Using cache dir: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64
2024-04-22 23:44:44 DEBUG LibUtils:67 - Loading huggingface library from: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64
2024-04-22 23:44:44 DEBUG LibUtils:71 - Loading native library: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64/libtokenizers.so
2024-04-22 23:44:47 INFO  GPT2GenerateText:31 - DeepMind Company is a global leader in the field of artificial intelligence and artificial intelligence. We are a leading provider of advanced AI solutions for the automotive industry, including the latest in advanced AI solutions for the automotive industry. We are also a leading provider of advanced AI solutions for the automotive industry, including the
2024-04-22 23:44:47 DEBUG DefaultModelZoo:67 - Scanning models in repo: class ai.djl.repository.SimpleUrlRepository, https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip
2024-04-22 23:44:47 DEBUG ModelZoo:111 - Loading model with Criteria:
	Application: UNDEFINED
	Input: class ai.djl.ndarray.NDList
	Output: class ai.djl.modality.nlp.generate.CausalLMOutput
	Engine: PyTorch
	ModelZoo: ai.djl.localmodelzoo

2024-04-22 23:44:47 DEBUG ModelZoo:115 - Searching model in specified model zoo: ai.djl.localmodelzoo
2024-04-22 23:44:47 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.localmodelzoo:gpt2_pt UNDEFINED [
	ai.djl.localmodelzoo/gpt2_pt/gpt2_pt {}
]
2024-04-22 23:44:47 DEBUG MRL:267 - Preparing artifact: https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip, ai.djl.localmodelzoo/gpt2_pt/gpt2_pt {}
2024-04-22 23:44:47 DEBUG AbstractRepository:126 - Files have been downloaded already: /home/diego/.djl.ai/cache/repo/model/undefined/ai/djl/localmodelzoo/5db362b8f477d219031fab733962dcf4407f55dc
2024-04-22 23:44:47 DEBUG JniUtils:1739 - mapLocation: false
2024-04-22 23:44:47 DEBUG JniUtils:1740 - extraFileKeys: []
2024-04-22 23:44:48 INFO  DeferredTranslatorFactory:68 - Using TranslatorFactory: ai.djl.pytorch.zoo.nlp.textgeneration.PtGptTranslatorFactory
2024-04-22 23:44:51 INFO  GPT2GenerateText:34 - DeepMind Company is a global leader in the field of artificial intelligence and artificial intelligence research and development.

Our mission is to provide the world with the best and brightest minds in the field of artificial intelligence and artificial intelligence research and development.

Our mission is to
```

##### Q/A
BertQaInference
```
2024-04-22 23:08:19 INFO  BertQaInference:34 - Paragraph: BBC Japan was a general entertainment Channel. Which operated between December 2004 and April 2006. It ceased operations after its Japanese distributor folded.
2024-04-22 23:08:19 INFO  BertQaInference:35 - Question: When did BBC Japan start broadcasting?
2024-04-22 23:08:19 DEBUG Engine:165 - Registering EngineProvider: Python
2024-04-22 23:08:19 DEBUG Engine:165 - Registering EngineProvider: MPI
2024-04-22 23:08:19 DEBUG Engine:165 - Registering EngineProvider: DeepSpeed
2024-04-22 23:08:19 DEBUG Engine:165 - Registering EngineProvider: PyTorch
2024-04-22 23:08:19 DEBUG Engine:165 - Registering EngineProvider: OnnxRuntime
2024-04-22 23:08:19 DEBUG Engine:95 - Found default engine: PyTorch
2024-04-22 23:08:22 DEBUG ModelZoo:111 - Loading model with Criteria:
	Application: NLP.QUESTION_ANSWER
	Input: class ai.djl.modality.nlp.qa.QAInput
	Output: class java.lang.String
	Engine: PyTorch
	No translator supplied

2024-04-22 23:08:22 DEBUG ModelZoo:138 - Ignore ModelZoo ai.djl.onnxruntime by engine: PyTorch
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:hf-internal-testing/tiny-bert-for-token-classification NLP.TOKEN_CLASSIFICATION [
	ai.djl.huggingface.pytorch/hf-internal-testing/tiny-bert-for-token-classification/0.0.1/tiny-bert-for-token-classification {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:hf-internal-testing/tiny-bert-for-token-classification
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking/0.0.1/distilbert-multilingual-nli-stsb-quora-ranking {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sonoisa/sentence-bert-base-ja-en-mean-tokens NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sonoisa/sentence-bert-base-ja-en-mean-tokens/0.0.1/sentence-bert-base-ja-en-mean-tokens {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sonoisa/sentence-bert-base-ja-en-mean-tokens
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/multi-qa-distilbert-dot-v1 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/multi-qa-distilbert-dot-v1/0.0.1/multi-qa-distilbert-dot-v1 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/multi-qa-distilbert-dot-v1
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:setu4993/LEALLA-large NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/setu4993/LEALLA-large/0.0.1/LEALLA-large {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:setu4993/LEALLA-large
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v4 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/msmarco-distilbert-base-v4/0.0.1/msmarco-distilbert-base-v4 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v4
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v3 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/msmarco-distilbert-base-v3/0.0.1/msmarco-distilbert-base-v3 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v3
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v2 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/msmarco-distilbert-base-v2/0.0.1/msmarco-distilbert-base-v2 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/msmarco-distilbert-base-v2
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/all-mpnet-base-v2 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/all-mpnet-base-v2/0.0.1/all-mpnet-base-v2 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/all-mpnet-base-v2
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:Seznam/dist-mpnet-paracrawl-cs-en NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/Seznam/dist-mpnet-paracrawl-cs-en/0.0.1/dist-mpnet-paracrawl-cs-en {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:Seznam/dist-mpnet-paracrawl-cs-en
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/all-mpnet-base-v1 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/sentence-transformers/all-mpnet-base-v1/0.0.1/all-mpnet-base-v1 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:sentence-transformers/all-mpnet-base-v1
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:FajnyKarton/st-polish-kartonberta-base-alpha-v1 NLP.TEXT_EMBEDDING [
	ai.djl.huggingface.pytorch/FajnyKarton/st-polish-kartonberta-base-alpha-v1/0.0.1/st-polish-kartonberta-base-alpha-v1 {"en":"true"}
]
2024-04-22 23:08:22 DEBUG ModelZoo:164 - application mismatch for ModelLoader: ai.djl.huggingface.pytorch:FajnyKarton/st-polish-kartonberta-base-alpha-v1
2024-04-22 23:08:22 DEBUG ModelZoo:151 - Checking ModelLoader: ai.djl.huggingface.pytorch:bert-large-uncased-whole-word-masking-finetuned-squad NLP.QUESTION_ANSWER [
	ai.djl.huggingface.pytorch/bert-large-uncased-whole-word-masking-finetuned-squad/0.0.1/bert-large-uncased-whole-word-masking-finetuned-squad {"en":"true"}
]
2024-04-22 23:08:22 DEBUG MRL:267 - Preparing artifact: Huggingface, ai.djl.huggingface.pytorch/bert-large-uncased-whole-word-masking-finetuned-squad/0.0.1/bert-large-uncased-whole-word-masking-finetuned-squad {"en":"true"}
2024-04-22 23:08:22 DEBUG AbstractRepository:126 - Files have been downloaded already: /home/diego/.djl.ai/cache/repo/model/nlp/question_answer/ai/djl/huggingface/pytorch/bert-large-uncased-whole-word-masking-finetuned-squad/true/0.0.1
Loading:     100% |████████████████████████████████████████|
2024-04-22 23:08:22 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:08:22 DEBUG LibUtils:410 - Using cache dir: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64
2024-04-22 23:08:22 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libgomp-52f2fd74.so.1
2024-04-22 23:08:22 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libc10.so
2024-04-22 23:08:22 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libtorch_cpu.so
2024-04-22 23:08:23 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/libtorch.so
2024-04-22 23:08:23 DEBUG LibUtils:374 - Loading native library: /home/diego/.djl.ai/pytorch/2.1.1-cpu-linux-x86_64/0.27.0-libdjl_torch.so
2024-04-22 23:08:23 INFO  PtEngine:67 - PyTorch graph executor optimizer is enabled, this may impact your inference latency and throughput. See: https://docs.djl.ai/docs/development/inference_performance_optimization.html#graph-executor-optimization
2024-04-22 23:08:23 INFO  PtEngine:72 - Number of inter-op threads is 6
2024-04-22 23:08:23 INFO  PtEngine:73 - Number of intra-op threads is 6
2024-04-22 23:08:23 DEBUG JniUtils:1739 - mapLocation: true
2024-04-22 23:08:23 DEBUG JniUtils:1740 - extraFileKeys: []
2024-04-22 23:08:24 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:08:24 DEBUG CudaUtils:76 - No GPU device found: no CUDA-capable device is detected (100)
2024-04-22 23:08:24 INFO  Platform:82 - Found matching platform from: jar:file:/home/diego/.m2/repository/ai/djl/huggingface/tokenizers/0.27.0/tokenizers-0.27.0.jar!/native/lib/tokenizers.properties
2024-04-22 23:08:24 DEBUG LibUtils:88 - Using cache dir: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64
2024-04-22 23:08:24 DEBUG LibUtils:67 - Loading huggingface library from: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64
2024-04-22 23:08:24 DEBUG LibUtils:71 - Loading native library: /home/diego/.djl.ai/tokenizers/0.15.0-0.27.0-linux-x86_64/libtokenizers.so
2024-04-22 23:08:25 INFO  BertQaInference:23 - Answer: december 2004
```