import torch
import functools
from pyhorn.data_processing import *

class Trainer:
    
    def __init__(self, loss, optimizer):
        """
        Object of Train class will train choosen model.
        
        --------------------------
        -- loss - a loss function (metric) to calculate error value on each iteration
        -- optimizer - a function to optimize model parameters such as model weights
        --------------------------

        """
        self.loss = loss
        self.optimizer = optimizer

    def train(self, net, *data, **options):
        """
        Train method for Neural Network

        --------------------------
        -- net - model to train
        -- epochs - number of epochs while training
        -- batch_size - subdata with batch_size len for training
        -- early_stopping - model stop training if quality on test data do not increase in {early_stopping} steps
        --------------------------

        """
        min_error = 1e6
        data = list(map(lambda x: to_cuda_tensor(x), data))
        for _ in range(options['epochs']):
            if 'batch_size' in options.keys(): self._train_on_batch(net, *data, **options)
            else: self._train_on_data(net, *data, **options)
            if 'early_stopping' in options.keys(): 
                self._early_stopping(net, min_error, *data, **options)
                if options['early_stopping'] == 0: break

    def _train_on_data(self, net, *data, **options):
        self.optimizer.zero_grad()
        prediction = net.forward(data[0])
        error = self.loss(prediction, data[1])
        error.backward()
        self.optimizer.step()
        return error

    def _train_on_batch(self, net, *data, **options):
        max_size = data[0].size()[0]
        batch_size = options['batch_size']
        for start_index in range(0, max_size, batch_size):
            end_index = start_index + batch_size
            error = self.train_on_data(self, net, data[0][start_index:end_index], data[1][start_index:end_index], **options)
        return error

    def early_stop(self, net, min_error,  *data, **options):
        prediction = net.forward(data[0])
        error = self.loss(prediction, data[1])
        if min_error > error: options['early_stopping'] -= 1
        else: net.best_parameters = net.parameters        