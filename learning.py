import torch
import os

class Extenssion:
    def __init__(self, stop_count, loss):
        self.stop_count = stop_count
        self.loss = loss
        self.min_loss = None
        self.best_parameters = None
    
    def stop(self, net, data, target):
        loss = self.loss(net.predict(data), target)
        if not self.min_loss: self.min_loss = loss
        if loss > self.min_loss: self.stop_count -= 1
        else: self.min_loss = loss; self.best_parameters = net.parameters
        return True if self.stop_count == 0 else False
    
    def save_weights(self, net, filepath = None):
        if filepath: torch.save(net, filepath)
        else:  torch.save(net, os.getcwd())

class Trainer:
    def __init__(self, options):
        self.loss = options['loss']
        self.optimizer = options['optimizer']
        
    def train(self, net, train_data, train_target, parameters):
        if 'batch_size' in parameters.keys(): return self._train_on_batch(net, train_data, train_target, parameters)
        else: return self._train_on_data(net, train_data, train_target)
    
    def _train_on_data(self, net, data_train, target_train):
        prediction = net.forward(data_train)
        residuals = self.loss(prediction, target_train)
        residuals.backward()
        self.optimizer.step()
        return net, residuals
        
    def _train_on_batch(self, net, train_data, train_target, parameters):
        max_size = train_data.size()[0]
        for start_index in range(0, max_size, parameters['batch_size']):
            end_index = start_index + parameters['batch_size']
            net, residuals = self._train_on_data(net, train_data[start_index:end_index], train_target[start_index:end_index])
        return net, residuals