# RNN - Neural Network

Predict the country where you name come from. Using `PyTorch` and `Neural Network`.

## How it works?

1. A Char-level recurrent neutal network(RNN) to classify words with Natual Language Processing (NLP).
2. A Char-level RNN reads words as series of chars as input and output to "Hidden State" at each step.
3. Chaining each hiden state to the next step.
4. The result is a final prediction.

`data_preparation.py`

5. Tousands of lastnames are trained across 18 languages(english, portuguese, italian, spanish, more...)
6. Training data is on the `data/[language].txt` files. Each files has a bunch of last names that need to be converted from Unicode to ASCII.
7. We will build a dictionary of a list of names(lines) per language(category).
8. A category is a language. The variable `all_categories` is all languages.
9. We have all names organized(in dictionaries) we need to turn them into `Tensors`
10. In order to represent a single letter, we use "one-hot vector" of size = `< 1 * num_letters >`
11. To make a word we will have a bunch of those into a 2D matrix = `< line_length * 1 * num_letters >`
12. Here is a tensor sample - as you can see is just a 2D matrix:
```python
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
```

`model.py`

13. Now we will create a Neural Network model(in a python class `model.py`).
14. Creating a neural network means cloning parameters of a layers over several timesteps.
15. The layers hold hidden state and gradients(measures the change in all weights with regard to the change in error.)
16. Our RNN model is:
    * 2 Linear Layers (operating on input and hidden state)
    * SoftMax layer after the output (The softmax function, also known as softargmax or normalized exponential function, converts a vector of K real numbers into a probability distribution of K possible outcomes)

`train.py`

17. Define some helper functions, we need to understand the output of the neural network(likelyhood of each category(language)).
18. Doing `Tensor.topk` to get the index of the greatest value.
18. There is another help function to get a sample of name/language.
19. Now we train the neural network
    * We just need a bunch of examples to train the network.
    * Make the network make guesses and tell if is right or wrong.
    * Loss function used, because we are using softmax will be: `nn.LLLoss`
20. For each training loop(alo know as epoch):
   * Create input and target tensors
   * Create zeroed initial hidden state (`torch.zeros(1, n_hidden)`)
   * Read each letter in and keep the state of the hidden state for next letter
   * Compare final output to target
   * Back-propagation: Backpropagation is a gradient estimation method used to train neural network models. The gradient estimate is used by the optimization algorithm to compute the network parameter updates.
   * Return output and loss
21. While training losses show start high and accurency low and should end in reverse(low loss and high accurancy).
22. Finally the model is saved.

`predict.py`

23. First of load we load the model. 
24. `predict` function recives a lastname and the number of categorios you can get(max 18).
25. The lastname is transformed to a tensor. 
26. Finaly we run a `topk` on the tensor to get the results, grouping cateogies and scores.

`server.py`

27. expose the prediction function as a REST endpoint.
28. call predict based on a rest parameter(lastname).

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

[X] 0. Make it work on the Jupyter notebook. <BR/>
[X] 1. Make it work on monolithic python code. <BR/>
[x] 2. Refactor the code to split between training and prediction <BR/>
[x] 3. Added tests <BR/>
[x] 4. bettter document the code <BR/>
