import torch
import torchvision
import torch.nn as nn
import numpy as np



if __name__ == '__main__':
    print('--------------------------')
    import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    import matplotlib.pyplot as plt


    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


    def relu(x):
        return np.maximum(0, x)


    def tanh(x):
        return np.tanh(x)


    sigmoid_inputs = np.arange(-10, 10)
    sigmoid_outputs = tanh(sigmoid_inputs)
    print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
    print("Sigmoid Function Output :: {}".format(sigmoid_outputs))

    plt.plot(sigmoid_inputs, sigmoid_outputs)
    # plt.xlabel("Sigmoid Inputs")
    # plt.ylabel("Sigmoid Outputs")
    plt.show()