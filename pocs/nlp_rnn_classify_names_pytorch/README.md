### Train
```
❯ ./train.sh
['data/names/Greek.txt', 'data/names/Scottish.txt', 'data/names/Polish.txt', 'data/names/Portuguese.txt', 'data/names/Dutch.txt', 'data/names/Vietnamese.txt', 'data/names/Italian.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Korean.txt', 'data/names/Spanish.txt', 'data/names/Russian.txt', 'data/names/German.txt', 'data/names/Irish.txt', 'data/names/Czech.txt', 'data/names/Japanese.txt', 'data/names/Arabic.txt', 'data/names/French.txt']
category = Russian / line = Mihel
category = Scottish / line = Jackson
category = English / line = Toal
category = Spanish / line = Silva
category = Scottish / line = Kennedy
category = Polish / line = Filipek
category = Scottish / line = Davidson
category = Czech / line = Vrazel
category = Irish / line = Mullen
category = Italian / line = Viola
5000 5% (0m 11s) 2.0447 Dickson / Scottish ✓
10000 10% (0m 26s) 1.0623 Nosek / Polish ✓
15000 15% (0m 38s) 0.0579 Vassilopulos / Greek ✓
20000 20% (0m 50s) 0.4614 Pereira / Portuguese ✓
25000 25% (1m 2s) 2.2693 Buchholz / Irish ✗ (German)
30000 30% (1m 14s) 0.9878 Karlovsky / Czech ✓
35000 35% (1m 26s) 1.0173 Do / Vietnamese ✓
40000 40% (1m 39s) 2.2604 Araya / Japanese ✗ (Spanish)
45000 45% (1m 51s) 1.9213 Hancock / German ✗ (English)
50000 50% (2m 4s) 1.7765 Santana / Spanish ✗ (Portuguese)
55000 55% (2m 18s) 1.8158 Sullivan / English ✗ (Irish)
60000 60% (2m 31s) 2.4558 Guirguis / French ✗ (Arabic)
65000 65% (2m 43s) 0.1783 Mao / Chinese ✓
70000 70% (2m 56s) 1.1213 Moreno / Italian ✗ (Portuguese)
75000 75% (3m 10s) 1.4540 O'Boyle / Russian ✗ (Irish)
80000 80% (3m 23s) 4.2319 De la fuente / French ✗ (Spanish)
85000 85% (3m 35s) 0.0111 Bilias / Greek ✓
90000 90% (3m 47s) 1.6929 Machado / Italian ✗ (Portuguese)
95000 95% (3m 59s) 0.1560 Hishida / Japanese ✓
100000 100% (4m 10s) 0.2872 Miyamoto / Japanese ✓
```

### Predict
```
❯ ./predict.sh
['data/names/Greek.txt', 'data/names/Scottish.txt', 'data/names/Polish.txt', 'data/names/Portuguese.txt', 'data/names/Dutch.txt', 'data/names/Vietnamese.txt', 'data/names/Italian.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Korean.txt', 'data/names/Spanish.txt', 'data/names/Russian.txt', 'data/names/German.txt', 'data/names/Irish.txt', 'data/names/Czech.txt', 'data/names/Japanese.txt', 'data/names/Arabic.txt', 'data/names/French.txt']

> Dovesky
(-2.76) Korean
(-2.80) English
(-2.82) Scottish

> Jackson
(-2.75) Korean
(-2.82) Irish
(-2.84) English

> Satoshi
(-2.79) Korean
(-2.81) English
(-2.83) Scottish

> Silva
(-2.73) Korean
(-2.79) English
(-2.82) Scottish

> Salvatore
(-2.79) Korean
(-2.81) English
(-2.82) Vietnamese

> Pacheco
(-2.77) Korean
(-2.79) Scottish
(-2.82) Vietnamese
```

### Run the app
```
./install-deps.sh
./run.sh
```
```
http://localhost:8080/silva
```
```
// 20240403015122
// http://localhost:8080/silva

{
  "result": [
    [
      -2.79306960105896,
      "Spanish"
    ],
    [
      -2.79726505279541,
      "Polish"
    ],
    [
      -2.8278114795684814,
      "Vietnamese"
    ],
    [
      -2.8366286754608154,
      "Dutch"
    ],
    [
      -2.8588709831237793,
      "Russian"
    ],
    [
      -2.861292839050293,
      "Korean"
    ],
    [
      -2.8627049922943115,
      "Italian"
    ],
    [
      -2.887091636657715,
      "French"
    ],
    [
      -2.898505449295044,
      "Arabic"
    ],
    [
      -2.898899793624878,
      "English"
    ]
  ]
}
```

### TODO

1. make the code work outside of the notebook
2. refactor the code to split between training and prediction
3. Added tests
4. bettter tocument the code

