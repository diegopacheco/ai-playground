# Llama cpp

## Install

```
curl -LsSf https://llama.app/install.sh | sh
```

## Serve Qween Model

```
llama serve -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q8_0
```

## Run

```
❯ llama serve -hf ggml-org/Qwen3.6-27B-GGUF:Q8_0 --port 8081
0.00.455.401 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.455.687 W srv  llama_server: -----------------
0.00.455.690 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.455.690 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.455.690 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.455.691 W srv  llama_server: -----------------
0.00.457.189 I srv    load_model: loading model 'ggml-org/Qwen3.6-27B-GGUF:Q8_0'
0.02.838.219 W load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
0.02.838.223 W load_hparams: if you encounter problems with accuracy, try adding --image-min-tokens 1024
0.02.838.223 W load_hparams: more info: https://github.com/ggml-org/llama.cpp/issues/16842

0.03.084.744 I srv    load_model: loaded multimodal model, '/Users/diegopacheco/.cache/huggingface/hub/models--ggml-org--Qwen3.6-27B-GGUF/snapshots/8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8/mmproj-Qwen3.6-27B-Q8_0.gguf'
0.03.256.236 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'true'
0.03.265.355 I srv          init: chat template supports preserving reasoning, consider enabling it via --reasoning-preserve
0.03.265.360 I srv  llama_server: model loaded
0.03.265.362 I srv  llama_server: listening on http://127.0.0.1:8081
1.03.217.163 I slot get_availabl: id  3 | task -1 | selected slot by LRU, t_last = -1
1.03.217.189 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0
1.09.879.264 I slot print_timing: id  3 | task 0 | n_decoded =    100, tg =  15.79 t/s, tg_3s =  15.79 t/s
1.12.882.980 I slot print_timing: id  3 | task 0 | n_decoded =    147, tg =  15.74 t/s, tg_3s =  15.65 t/s
1.15.898.284 I slot print_timing: id  3 | task 0 | n_decoded =    194, tg =  15.71 t/s, tg_3s =  15.59 t/s
1.18.924.382 I slot print_timing: id  3 | task 0 | n_decoded =    241, tg =  15.67 t/s, tg_3s =  15.53 t/s
1.21.972.002 I slot print_timing: id  3 | task 0 | n_decoded =    288, tg =  15.63 t/s, tg_3s =  15.42 t/s
1.25.020.256 I slot print_timing: id  3 | task 0 | n_decoded =    335, tg =  15.60 t/s, tg_3s =  15.42 t/s
1.28.023.050 I slot print_timing: id  3 | task 0 | n_decoded =    381, tg =  15.57 t/s, tg_3s =  15.32 t/s
1.31.032.276 I slot print_timing: id  3 | task 0 | n_decoded =    427, tg =  15.54 t/s, tg_3s =  15.29 t/s
1.34.044.585 I slot print_timing: id  3 | task 0 | n_decoded =    473, tg =  15.51 t/s, tg_3s =  15.27 t/s
1.36.602.803 I slot print_timing: id  3 | task 0 | prompt eval time =     328.88 ms /    17 tokens (   19.35 ms per token,    51.69 tokens per second)
1.36.602.805 I slot print_timing: id  3 | task 0 |        eval time =   33056.29 ms /   512 tokens (   64.56 ms per token,    15.49 tokens per second)
1.36.602.806 I slot print_timing: id  3 | task 0 |       total time =   33385.18 ms /   529 tokens
1.36.602.806 I slot print_timing: id  3 | task 0 |    graphs reused =        509
1.36.602.821 I slot      release: id  3 | task 0 | stop processing: n_tokens = 528, truncated = 0
3.45.080.919 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.032
3.45.116.642 I slot launch_slot_: id  3 | task 514 | processing task, is_child = 0
3.51.596.091 I slot print_timing: id  3 | task 514 | n_decoded =    100, tg =  15.77 t/s, tg_3s =  15.77 t/s
3.54.611.669 I slot print_timing: id  3 | task 514 | n_decoded =    147, tg =  15.71 t/s, tg_3s =  15.59 t/s
3.57.645.436 I slot print_timing: id  3 | task 514 | n_decoded =    194, tg =  15.66 t/s, tg_3s =  15.49 t/s
4.00.690.939 I slot print_timing: id  3 | task 514 | n_decoded =    241, tg =  15.61 t/s, tg_3s =  15.43 t/s
4.03.743.974 I slot print_timing: id  3 | task 514 | n_decoded =    288, tg =  15.58 t/s, tg_3s =  15.39 t/s
4.06.750.059 I slot print_timing: id  3 | task 514 | n_decoded =    334, tg =  15.54 t/s, tg_3s =  15.30 t/s
4.09.763.676 I slot print_timing: id  3 | task 514 | n_decoded =    380, tg =  15.51 t/s, tg_3s =  15.26 t/s
4.12.797.144 I slot print_timing: id  3 | task 514 | n_decoded =    426, tg =  15.47 t/s, tg_3s =  15.16 t/s
4.15.837.740 I slot print_timing: id  3 | task 514 | n_decoded =    472, tg =  15.43 t/s, tg_3s =  15.13 t/s
4.18.503.672 I slot print_timing: id  3 | task 514 | prompt eval time =     138.94 ms /     4 tokens (   34.73 ms per token,    28.79 tokens per second)
4.18.503.675 I slot print_timing: id  3 | task 514 |        eval time =   33247.67 ms /   512 tokens (   64.94 ms per token,    15.40 tokens per second)
4.18.503.676 I slot print_timing: id  3 | task 514 |       total time =   33386.61 ms /   516 tokens
4.18.503.676 I slot print_timing: id  3 | task 514 |    graphs reused =       1017
4.18.503.697 I slot      release: id  3 | task 514 | stop processing: n_tokens = 528, truncated = 0
4.34.970.386 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 0.214 (> 0.100 thold), f_keep = 0.006
4.34.996.145 I slot launch_slot_: id  3 | task 1027 | processing task, is_child = 0
4.41.655.976 I slot print_timing: id  3 | task 1027 | n_decoded =    100, tg =  15.76 t/s, tg_3s =  15.76 t/s
4.44.707.274 I slot print_timing: id  3 | task 1027 | n_decoded =    147, tg =  15.64 t/s, tg_3s =  15.40 t/s
4.47.719.481 I slot print_timing: id  3 | task 1027 | n_decoded =    193, tg =  15.55 t/s, tg_3s =  15.27 t/s
4.48.116.154 I slot print_timing: id  3 | task 1027 | prompt eval time =     313.86 ms /    14 tokens (   22.42 ms per token,    44.61 tokens per second)
4.48.116.156 I slot print_timing: id  3 | task 1027 |        eval time =   12805.73 ms /   199 tokens (   64.35 ms per token,    15.54 tokens per second)
4.48.116.157 I slot print_timing: id  3 | task 1027 |       total time =   13119.59 ms /   213 tokens
4.48.116.157 I slot print_timing: id  3 | task 1027 |    graphs reused =       1214
4.48.116.169 I slot      release: id  3 | task 1027 | stop processing: n_tokens = 212, truncated = 0
5.12.612.314 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 0.176 (> 0.100 thold), f_keep = 0.014
5.12.629.812 I slot launch_slot_: id  3 | task 1228 | processing task, is_child = 0
5.19.285.784 I slot print_timing: id  3 | task 1228 | n_decoded =    100, tg =  15.77 t/s, tg_3s =  15.77 t/s
5.22.332.921 I slot print_timing: id  3 | task 1228 | n_decoded =    147, tg =  15.66 t/s, tg_3s =  15.42 t/s
5.25.350.502 I slot print_timing: id  3 | task 1228 | n_decoded =    193, tg =  15.56 t/s, tg_3s =  15.24 t/s
5.28.408.244 I slot print_timing: id  3 | task 1228 | n_decoded =    239, tg =  15.46 t/s, tg_3s =  15.04 t/s
5.31.458.191 I slot print_timing: id  3 | task 1228 | n_decoded =    284, tg =  15.34 t/s, tg_3s =  14.75 t/s
5.34.480.183 I slot print_timing: id  3 | task 1228 | n_decoded =    328, tg =  15.23 t/s, tg_3s =  14.56 t/s
5.37.522.728 I slot print_timing: id  3 | task 1228 | n_decoded =    372, tg =  15.14 t/s, tg_3s =  14.46 t/s
5.40.584.370 I slot print_timing: id  3 | task 1228 | n_decoded =    416, tg =  15.05 t/s, tg_3s =  14.37 t/s
5.43.610.014 I slot print_timing: id  3 | task 1228 | n_decoded =    455, tg =  14.84 t/s, tg_3s =  12.89 t/s
5.46.685.888 I slot print_timing: id  3 | task 1228 | n_decoded =    492, tg =  14.58 t/s, tg_3s =  12.03 t/s
5.49.706.019 I slot print_timing: id  3 | task 1228 | n_decoded =    526, tg =  14.31 t/s, tg_3s =  11.26 t/s
5.52.722.390 I slot print_timing: id  3 | task 1228 | n_decoded =    558, tg =  14.03 t/s, tg_3s =  10.61 t/s
5.55.748.504 I slot print_timing: id  3 | task 1228 | n_decoded =    588, tg =  13.74 t/s, tg_3s =   9.91 t/s
5.58.851.960 I slot print_timing: id  3 | task 1228 | n_decoded =    617, tg =  13.44 t/s, tg_3s =   9.34 t/s
6.01.916.022 I slot print_timing: id  3 | task 1228 | n_decoded =    646, tg =  13.19 t/s, tg_3s =   9.46 t/s
6.04.975.216 I slot print_timing: id  3 | task 1228 | n_decoded =    676, tg =  12.99 t/s, tg_3s =   9.81 t/s
6.08.034.645 I slot print_timing: id  3 | task 1228 | n_decoded =    707, tg =  12.83 t/s, tg_3s =  10.13 t/s
6.11.090.473 I slot print_timing: id  3 | task 1228 | n_decoded =    739, tg =  12.71 t/s, tg_3s =  10.47 t/s
6.12.839.700 W srv          stop: cancel task, id_task = 1228
6.12.862.548 I slot      release: id  3 | task 1228 | stop processing: n_tokens = 774, truncated = 0
6.26.214.786 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 0.842 (> 0.100 thold), f_keep = 0.021
6.26.233.703 I slot launch_slot_: id  3 | task 1989 | processing task, is_child = 0
6.34.664.344 I slot print_timing: id  3 | task 1989 | n_decoded =    100, tg =  12.16 t/s, tg_3s =  12.16 t/s
6.35.793.307 I slot print_timing: id  3 | task 1989 | prompt eval time =     205.90 ms /     6 tokens (   34.32 ms per token,    29.14 tokens per second)
6.35.793.309 I slot print_timing: id  3 | task 1989 |        eval time =    9353.29 ms /   113 tokens (   82.77 ms per token,    12.08 tokens per second)
6.35.793.310 I slot print_timing: id  3 | task 1989 |       total time =    9559.18 ms /   119 tokens
6.35.793.310 I slot print_timing: id  3 | task 1989 |    graphs reused =       2078
6.35.793.322 I slot      release: id  3 | task 1989 | stop processing: n_tokens = 131, truncated = 0
7.10.152.473 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.145
7.10.184.550 I slot launch_slot_: id  3 | task 2104 | processing task, is_child = 0
7.18.318.862 I slot print_timing: id  3 | task 2104 | n_decoded =    100, tg =  12.50 t/s, tg_3s =  12.50 t/s
7.19.127.071 I slot print_timing: id  3 | task 2104 | prompt eval time =     131.66 ms /     4 tokens (   32.91 ms per token,    30.38 tokens per second)
7.19.127.074 I slot print_timing: id  3 | task 2104 |        eval time =    8810.44 ms /   110 tokens (   80.09 ms per token,    12.49 tokens per second)
7.19.127.074 I slot print_timing: id  3 | task 2104 |       total time =    8942.10 ms /   114 tokens
7.19.127.074 I slot print_timing: id  3 | task 2104 |    graphs reused =       2186
7.19.127.086 I slot      release: id  3 | task 2104 | stop processing: n_tokens = 128, truncated = 0
8.17.092.617 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.148
8.17.113.029 I slot launch_slot_: id  3 | task 2215 | processing task, is_child = 0
8.23.575.151 I slot print_timing: id  3 | task 2215 | n_decoded =    100, tg =  15.83 t/s, tg_3s =  15.83 t/s
8.23.957.109 I slot print_timing: id  3 | task 2215 | prompt eval time =     145.90 ms /     4 tokens (   36.48 ms per token,    27.42 tokens per second)
8.23.957.112 I slot print_timing: id  3 | task 2215 |        eval time =    6697.81 ms /   106 tokens (   63.19 ms per token,    15.83 tokens per second)
8.23.957.112 I slot print_timing: id  3 | task 2215 |       total time =    6843.71 ms /   110 tokens
8.23.957.112 I slot print_timing: id  3 | task 2215 |    graphs reused =       2290
8.23.957.122 I slot      release: id  3 | task 2215 | stop processing: n_tokens = 124, truncated = 0
```

```
❯   ./run.sh "Write a short poem about C++"
```
```
In curly braces, the logic binds,
Where pointers dance and memory winds.
A semicolon ends the line,
A sharp and precise design.

From `#include` to `main`’s command,
You walk the compiler’s strictest land.
With classes built and templates spread,
Efficiency is what we’ve said.

No garbage collector here to sweep,
But promises you must keep.
In every bug, a lesson learned,
In C++, the code is earned.
```