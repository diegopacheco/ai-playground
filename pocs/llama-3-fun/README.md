### Login on Huggingface
```
/bin/pip install huggingface-cli
huggingface-cli login
```
### Result

LLMA 3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```
❯ time ./run.sh
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
2024-04-19 03:55:10.861303: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-19 03:55:11.756507: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.23it/s]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Arrrr, shiver me timbers! Me name be Captain Chat, the scurviest pirate chatbot to ever sail the seven seas! Me be here to swab the decks of yer mind with me witty banter and me trusty knowledge o' the high seas! So hoist the colors, me hearty, and let's set sail fer a swashbucklin' good time!
./run.sh  825,96s user 12,00s system 135% cpu 10:16,97 total
```