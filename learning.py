import torch
import os

# class Extenssion:
#     def __init__(self, stop_count, loss):
#         self.stop_count = stop_count
#         self.loss = loss
#         self.min_loss = None
#         self.best_parameters = None
    
#     def stop(self, net, data, target):
#         loss = self.loss(net.predict(data), target)
#         if not self.min_loss: self.min_loss = loss
#         if loss > self.min_loss: self.stop_count -= 1
#         else: self.min_loss = loss; self.best_parameters = net.parameters
#         return True if self.stop_count == 0 else False
    
#     def save_weights(self, net, filepath = None):
#         if filepath: torch.save(net, filepath)
#         else:  torch.save(net, os.getcwd())