### Install ONNX cli
```
/bin/pip install optimum onnx onnxruntime
```

### Result
```
Framework not specified. Using pt to export the model.
modules.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 3.17MB/s]
config_sentence_transformers.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 1.36MB/s]
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10.7k/10.7k [00:00<00:00, 79.7MB/s]
sentence_bert_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 624kB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 4.91MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 90.9M/90.9M [00:02<00:00, 37.8MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 350/350 [00:00<00:00, 3.95MB/s]
vocab.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 1.78MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 7.26MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 1.04MB/s]
1_Pooling/config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.50MB/s]
Automatic task detection to feature-extraction (possible synonyms are: default, image-feature-extraction, mask-generation, sentence-similarity).
Using the export variant default. Available variants are:
    - default: The default ONNX variant.

***** Exporting submodel 1/1: SentenceTransformer *****
Using framework PyTorch: 2.2.1+cu121
Overriding 1 configuration item(s)
        - use_cache -> False
Post-processing the exported models...
Deduplicating shared (tied) weights...

Validating ONNX model model.onnx...
        -[✓] ONNX model output names match reference model (token_embeddings, sentence_embedding)
        - Validating ONNX Model output "token_embeddings":
                -[✓] (2, 16, 384) matches (2, 16, 384)
                -[✓] all values close (atol: 1e-05)
        - Validating ONNX Model output "sentence_embedding":
                -[✓] (2, 384) matches (2, 384)
                -[✓] all values close (atol: 1e-05)
The ONNX export succeeded and the exported model was saved at: .
```