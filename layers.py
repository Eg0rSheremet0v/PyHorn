import torch

class Input(torch.nn.Module):
    def __init__(self, neurons_number):
        """
        The first layer of neural network. Needs to specify input shape.
        ____________________________________

        neurons_number - shape of input data

        """
        super(Input, self).__init__()
        self.type = 'Input'
        self.neurons_number = neurons_number

    def forward(self, input):
        return input

class Dense(torch.nn.Module):
    def __init__(self, neurons_number, activation, weights_initialize='rand_norm'):
        """
        The main layer of neural network.
        ____________________________________

        neurons_number - number of neurons in layer
        activation - function to calculate an output signal
        weights_initialize - method to precompute weights of the layer

        """
        super(Dense, self).__init__()
        self.type = 'Dense'
        self.neurons_number = neurons_number
        self.weights_initialize = weights_initialize
        self.activation = activation
        self.weights = None
        self.bias = None
    
    def init_weights(self, prev_layer_number = None):
        weights = torch.randn(prev_layer_number, self.neurons_number, dtype=torch.float)
        bias = torch.randn(self.neurons_number, dtype=torch.float)
        
        if self.weights_initialize == 'zero': weigths = torch.zeros((prev_layer_number, self.neurons_number), dtype=torch.float)
        elif self.weights_initialize == 'ones': weights = torch.ones((prev_layer_number, self.neurons_number), dtype=torch.float)
        elif self.weights_initialize == 'rand_xavier': torch.nn.init.xavier_uniform_(weights)
        elif self.weights_initialize == 'rand_he': weights = torch.nn.init.kaiming_uniform_(weights)
        
        weights.requires_grad_(True)
        bias.requires_grad_(True)
        
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(bias)
        
        self.register_parameter('weights', self.weights)
        self.register_parameter('bias', self.bias)
        
    def freeze(self):
        """
        Method to freeze the layer for learning.

        """
        self.weights.requires_grad_(False)
  
    def forward(self, input):
        signal = torch.matmul(input, self.weights) + self.bias
        return self.activation(signal)

class Dropout(torch.nn.Module):
    def __init__(self, proba):
        """
        Layer that applies a fraction rate of input neurons.
        ____________________________________

        proba - probability of input neurons activation (0 to 1)

        """
        super(Dropout, self).__init__()
        self.type = 'Dropout'
        self.proba = proba
        self.is_on = True
        self.neurons_number = None
        self.weights = None
  
    def forward(self, input):
        return input * self.weights.sample().cuda() * self.proba if self.is_on else input
  
    def init_weights(self, input_size):
        probabilities = torch.tensor([self.proba] * input_size, dtype=torch.float)
        self.weights = torch.distributions.Bernoulli(probabilities)
        self.neurons_number = input_size

class Norm(torch.nn.Module):
    def __init__(self):
        """
        Layer that applies norm function.
        """
        super(Norm, self).__init__()
        self.neurons_number = None
        self.weights = None
        self.bias = None
    
    def forward(self, input):
        input_norm = (input - torch.mean(input)) / torch.var(input)
        return input_norm * self.weights + self.bias
  
    def init_weights(self, input_size):
        self.neurons_number = input_size
        
        self.weights = torch.nn.Parameter(torch.ones(input_size, dtype=torch.float, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.zeros(input_size, dtype=torch.float, requires_grad=True))
        
        self.register_parameter('weights', self.weights)
        self.register_parameter('bias', self.bias)

class NALU(torch.nn.Module):
    def __init__(self, neurons_number, eps=1e-5):
        """
        Neural Arithmetic Logic Unit. The layer wich can learn arithmetical function with gate complex solution.
        ____________________________________

        neurons_number - number of neurons in the layer
        eps - coef of gate rate
        
        """
        super(NALU, self).__init__()
        self.neurons_number = neurons_number
        self.type = 'NALU'
        self.eps = eps
        self.weights_simple_tanh = None
        self.weights_simple_sigm = None
        self.weights_complex_tanh = None
        self.weights_complex_sigm = None
        self.gate = None

        
    def init_weights(self, input_size):
        self._init_weights_simple(input_size)
        self._init_weights_complex(input_size)
        
        self.gate = torch.nn.Parameter(torch.randn(input_size, self.neurons_number, dtype=torch.float, requires_grad=True))
        self.register_parameter('gate', self.gate)
        
    def _init_weights_simple(self, input_size):
        self.weights_simple_tanh = torch.nn.Parameter(torch.FloatTensor(input_size, self.neurons_number).uniform_(-2,2))
        self.weights_simple_sigm = torch.nn.Parameter(torch.FloatTensor(input_size, self.neurons_number).uniform_(-2,2))
        
        self.weights_simple_tanh.requires_grad_(True)
        self.weights_simple_sigm.requires_grad_(True)
        
        self.register_parameter('weights_simple_tanh', self.weights_simple_tanh)
        self.register_parameter('weights_simple_sigm', self.weights_simple_sigm)
        
    def _init_weights_complex(self, input_size):
        self.weights_complex_tanh = torch.nn.Parameter(torch.FloatTensor(input_size, self.neurons_number).uniform_(-2,2))
        self.weights_complex_sigm = torch.nn.Parameter(torch.FloatTensor(input_size, self.neurons_number).uniform_(-2,2))
        
        self.weights_complex_tanh.requires_grad_(True)
        self.weights_complex_sigm.requires_grad_(True)
        
        self.register_parameter('weights_complex_tanh', self.weights_complex_tanh)
        self.register_parameter('weights_complex_sigm', self.weights_complex_sigm)
        
    def forward(self, input):
        w_simple = torch.tanh(self.weights_simple_tanh) * torch.sigmoid(self.weights_simple_sigm)
        w_complex = torch.tanh(self.weights_complex_tanh) * torch.sigmoid(self.weights_complex_sigm)
        nac_s = torch.matmul(input, w_simple)
        nac_c = torch.exp(torch.matmul(torch.log(torch.abs(input) + self.eps), w_complex))
        g = torch.sigmoid(torch.matmul(input, self.gate))
        output = g * nac_s + (1 - g) * nac_c
        return output