### Whats going one here?

#### src/main.py
* Uses pytorce to download a dataset: FashionMNIST.
* FashionMNIST https://www.kaggle.com/datasets/zalando-research/fashionmnist has 60k examples 10k for testing.
* We are doing a NeuralNetwork.
* Data is being trained in 5 rounds (epochs)
* PyTorch has a concept of "device" I'm using CPU but could be GPU with CUDA.
* There is some prediction / evaluation going on
* Model is saved to disk in a pytorch format .pth which 
is a zig file with binary data on - model is not big ~2.7MB.

#### src/load-and-predict.py
* Loading the model from the file system (.pth file)
* Load the model into the NeuralNetwork
* Using the test data to check predictions
