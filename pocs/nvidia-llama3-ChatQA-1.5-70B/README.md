### Result
* LLM Model nvidia/Llama3-ChatQA-1.5-70B
* Question and Answer
* 70B parameters
* 1.5B tokens
* This model works by generating answers to questions based on the context of the question. The model is trained on a large dataset of question and answer pairs

### Output
```
 The percentage change of the net income from Q4 FY23 to Q4 FY24 is calculated using the formula ((12285 - 1414) / 1414 * 100), resulting in a 769% increase.
./run.sh  38500,81s user 2694,03s system 87% cpu 13:07:35,08 total
```
took 10.6h to generate the answer

### Downloads
```
/home/diego/.local/lib/python3.10/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50.6k/50.6k [00:00<00:00, 1.08MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.08M/9.08M [00:00<00:00, 16.6MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 73.0/73.0 [00:00<00:00, 110kB/s]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 653/653 [00:00<00:00, 857kB/s]
/home/diego/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:749: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
model.safetensors.index.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 69.9k/69.9k [00:00<00:00, 41.8MB/s]
model-00001-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.72G/9.72G [02:16<00:00, 71.4MB/s]
model-00002-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:14<00:00, 73.0MB/s]
model-00003-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.97G/9.97G [02:13<00:00, 74.8MB/s]
model-00004-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:10<00:00, 75.0MB/s]
model-00005-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:14<00:00, 72.8MB/s]
model-00006-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:12<00:00, 73.9MB/s]
model-00007-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.97G/9.97G [02:15<00:00, 73.7MB/s]
model-00008-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:13<00:00, 73.5MB/s]
model-00009-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:11<00:00, 74.2MB/s]
model-00010-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:14<00:00, 72.8MB/s]
model-00011-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.97G/9.97G [03:23<00:00, 48.9MB/s]
model-00012-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:14<00:00, 73.1MB/s]
model-00013-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:11<00:00, 74.3MB/s]
model-00014-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 9.80G/9.80G [02:12<00:00, 73.7MB/s]
model-00015-of-00015.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3.51G/3.51G [00:47<00:00, 73.3MB/s]
Downloading shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [33:11<00:00, 132.78s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:02<00:00,  7.18it/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136/136 [00:00<00:00, 194kB/s]
WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu and disk.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
```
