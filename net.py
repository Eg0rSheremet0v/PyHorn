import torch
from pyhorn.data_processing import *
from pyhorn.learning import *
from pyhorn.train import *

class Net(torch.nn.Sequential):
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

  def train(self, trainer, *data, **options):
    """
    Method to train the model.

    --------------------------
    -- trainer - trainer object to train the model
    -- epochs - number of epochs while training
    -- batch_size - subdata with batch_size len for training
    -- early_stopping - model stop training if quality on test data do not increase in {early_stopping} steps
    --------------------------

    """
    self._set_layer('Dropout', True)
    trainer.train(self, *data, **options)
    # if 'early_stop' in options.keys(): 
    #   data, test_data, target, test_target = make_holdout(data, target)
    # for epoch in range(options['epochs']):
      # if self.evaluater:
      #   if self.evaluater.stop(self, test_data, test_target): 
      #     self.parameters = self.evaluater.best_parameters
      #     print('early stop on: %i with loss_train: %f || loss_test: %f' % (epoch, residuals, self.evaluater.min_loss))
      #     break
        
  def freeze_layers(self, level):
    for layer in self.layers[:level]:
      if layer.type != 'Input': layer.freeze()   
  
  def _set_layer(self, layer_type, mark):
    for layer in self.layers:
      if layer.type == layer_type: layer.is_on = mark