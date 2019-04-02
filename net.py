import torch
from pyhorn.data_processing import *
from pyhorn.learning import *
from pyhorn.train import *

class Net(torch.nn.Sequential):

  best_parameters = None

  def __init__(self):
    super().__init__()
    self.number_of_layers = 0
    self.layers = torch.nn.ModuleList()
    self.trainer = None
    self.evaluater = None
 
  def append(self, layer):
    if layer.type != 'Input': layer.init_weights(self.layers[-1].neurons_count)
    self.layers.append(layer)
    
  def forward(self, input):
    for layer in self.layers:
      input = layer.forward(input)
    return input
  
  def predict(self, input):
    input = to_cuda_tensor(input)
    self._set_layer('Dropout', False)
    prediction = self.forward(input)
    self._set_layer('Dropout', True)
    return prediction

  # def learn(self, trainer, *data, **options):
    # self._set_layer('Dropout', True)
    # trainer.train(self, *data, **options)
    
        
  def freeze_layers(self, level):
    for layer in self.layers[:level]:
      if layer.type != 'Input': layer.freeze()   
  
  def _set_layer(self, layer_type, mark):
    for layer in self.layers:
      if layer.type == layer_type: layer.is_on = mark