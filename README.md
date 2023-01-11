# DL Framework
Deep learning framework implementation on pure NumPy

Implemented stackable layers:
* Linear
* Batch Normalization (including shift and scale phase)
* Dropout
* ReLU
* LeakyReLU, 
* ELU 
* SoftMax
* LogSoftMax 


Implemented forward/backward passage of this layers.

Implemented Cross Entropy Loss and Negative Log-Likelihood Loss.

Implemented SGD and ADAM optimizers.

Implemented early stopping and reducing learning rate on plateau.

**Tested of MNIST dataset**

The best result is `98.4%` accuracy on test (note that there were no convolutional layers)
