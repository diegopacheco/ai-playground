### Train
```
❯ python src/train.py
['data/names/Greek.txt', 'data/names/Scottish.txt', 'data/names/Polish.txt', 'data/names/Portuguese.txt', 'data/names/Dutch.txt', 'data/names/Vietnamese.txt', 'data/names/Italian.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Korean.txt', 'data/names/Spanish.txt', 'data/names/Russian.txt', 'data/names/German.txt', 'data/names/Irish.txt', 'data/names/Czech.txt', 'data/names/Japanese.txt', 'data/names/Arabic.txt', 'data/names/French.txt']
Slusarski
18
['Greek', 'Scottish', 'Polish', 'Portuguese', 'Dutch', 'Vietnamese', 'Italian', 'Chinese', 'English', 'Korean', 'Spanish', 'Russian', 'German', 'Irish', 'Czech', 'Japanese', 'Arabic', 'French']
/home/linuxbrew/.linuxbrew/Cellar/python@3.11/3.11.7_1/lib/python3.11/site-packages/torch/nn/modules/module.py:1511: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return self._call_impl(*args, **kwargs)
5000 5% (0m 15s) 3.1414 Thien / Irish ✗ (Chinese)
10000 10% (0m 33s) 3.8321 Laar / Vietnamese ✗ (Dutch)
15000 15% (0m 51s) 1.5032 Pho / Chinese ✗ (Vietnamese)
20000 20% (1m 7s) 1.8531 Fencl / German ✗ (Czech)
25000 25% (1m 25s) 1.2921 Cathan / Arabic ✗ (Irish)
30000 30% (1m 41s) 1.7108 Nicholl / Scottish ✗ (English)
35000 35% (1m 55s) 1.7326 Anisemenok / Polish ✗ (Russian)
40000 40% (2m 10s) 2.3952 Sze  / Korean ✗ (Chinese)
45000 45% (2m 25s) 0.0376 Jaskolski / Polish ✓
50000 50% (2m 41s) 1.1840 Allan / Scottish ✗ (English)
55000 55% (2m 56s) 2.8126 Simonis / Greek ✗ (Dutch)
60000 60% (3m 9s) 1.0732 Honjas / Greek ✓
65000 65% (3m 22s) 0.5973 Rocco / Italian ✓
70000 70% (3m 36s) 2.2929 Nonomura / Portuguese ✗ (Japanese)
75000 75% (3m 50s) 1.5516 Roma / Spanish ✗ (Italian)
80000 80% (4m 5s) 1.5903 Moles / Portuguese ✗ (Spanish)
85000 85% (4m 20s) 0.6485 Thi / Vietnamese ✓
90000 90% (4m 34s) 1.7810 Crespo / Portuguese ✗ (Spanish)
95000 95% (4m 49s) 0.1359 Zielinski / Polish ✓
100000 100% (5m 3s) 0.0256 Yamazaki / Japanese ✓
```

### Predict
```
python src/predict.py Hazaki
```
