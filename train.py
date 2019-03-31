import torch
import functools
from pyhorn.data_processing import *

class Extenssions:
    @staticmethod
    def train_on_batch(func):
        @functools.wraps(func)
        def wraper(self, net, *data, **options):
            max_size = data[0].size()[0]
            batch_size = options['batch_size']
            for start_index in range(0, max_size, batch_size):
                end_index = start_index + batch_size
                residuals = func(self, net, data[0][start_index:end_index], data[1][start_index:end_index], **options)
            return residuals
        return wraper


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
        data = list(map(lambda x: to_cuda_tensor(x), data))
        for _ in range(options['epochs']):
            self.train_on_data(net, *data, **options)

    @Extenssions.train_on_batch
    def train_on_data(self, net, *data, **options):
        self.optimizer.zero_grad()
        prediction = net.forward(data[0])
        residuals = self.loss(prediction, data[1])
        residuals.backward()
        self.optimizer.step()
        return residuals

