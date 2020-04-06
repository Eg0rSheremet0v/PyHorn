import torch
import os

class Extenssion:
    def __init__(self, stop_count, loss):
        self.stop_count = stop_count
        self.loss = loss
        self.min_loss = None
        self.best_parameters = None
    
    def stop(self, net, data, labels):
        """
        Method that stop a learning process (if specified).
        ____________________________________

        net - trainable neural network
        data - test input data
        labels - test target for input data

        """
        loss = self.loss(net.predict(data), labels)
        if not self.min_loss: self.min_loss = loss
        if loss > self.min_loss: self.stop_count -= 1
        else: self.min_loss = loss; self.best_parameters = net.parameters
        return True if self.stop_count == 0 else False
    
    def save_weights(self, net, filepath = None):
        """
        Method to save weights of all layers in the neural network.
        """
        if filepath: torch.save(net, filepath)
        else:  torch.save(net, os.getcwd())

class Trainer:
    def __init__(self, options):
        """
        Module to train neural networks.
        ____________________________________

        options - 
        """
        self.loss = options['loss']
        self.optimizer = options['optimizer']
        
    def train(self, net, train_data, train_labels, parameters):
        if 'batch_size' in parameters.keys(): return self._train_on_batch(net, train_data, train_labels, parameters)
        else: return self._train_on_data(net, train_data, train_labels)
    
    def _train_on_data(self, net, data_train, labels_train):
        prediction = net.forward(data_train)
        residuals = self.loss(prediction, labels_train)
        residuals.backward()
        self.optimizer.step()
        return net, residuals
        
    def _train_on_batch(self, net, train_data, train_labels, parameters):
        max_size = train_data.size()[0]
        for start_index in range(0, max_size, parameters['batch_size']):
            end_index = start_index + parameters['batch_size']
            net, residuals = self._train_on_data(net, train_data[start_index:end_index], train_labels[start_index:end_index])
        return net, residuals