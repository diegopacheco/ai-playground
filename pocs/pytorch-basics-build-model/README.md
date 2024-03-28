### Build Model?

* Neural netwroks have layers and modules.
* Neural netwroks have layers and modules perform operations in data.
* Namespace `torch.nn` provides abstractions to create your own neural network.
* A Neural Network is a module that consist of other modules (layers).
* Do not call model.forward() directly!
* FashionMNIST model sample:
  * nn.Flatten layer: convert each 2D 28x28 image into a array of 784 pixel 
  * nn.Linear layer: linear transformation on the input using stored weights and biases.
  * nn.ReLU: 
     * complex mappings between the model’s inputs and outputs
     * helping neural networks learn a wide variety of phenomena
  * nn.Sequential: ordered container of modules
    ```
        seq_modules = nn.Sequential(
            flatten,
            layer1,
            nn.ReLU(),
            nn.Linear(20, 10)
        )
    ```     
  * nn.Softmax: 
     * The last linear layer of the neural network returns logits
     * The logits are scaled to values [0, 1]   