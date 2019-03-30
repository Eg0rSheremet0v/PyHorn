import torch
from pyhorn.data_processing import *
from pyhorn.learning import *

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

  def train(self, data, target, **options):
    self._set_layer('Dropout', True) 
    self.loss = options['loss']
    self.optimizer = options['optimizer']
    data = to_cuda_tensor(data)
    target = to_cuda_tensor(target)
    if 'early_stop' in options.keys(): 
      self.evaluater = Extenssion(options['early_stop'], options['loss'])
      data, test_data, target, test_target = make_holdout(data, target)
    for epoch in range(options['epochs']):
      residuals = self._train_epoch(data, target, options)
      if self.evaluater:
        if self.evaluater.stop(self, test_data, test_target): 
          self.parameters = self.evaluater.best_parameters
          print('early stop on: %i with loss_train: %f || loss_test: %f' % (epoch, residuals, self.evaluater.min_loss))
          break
  
  def _train_epoch(self, data, target, options):
    if 'batch_size' in options.keys(): return self._train_on_batch( data, target, options['batch_size'])
    else: return self._train_on_full_data(data, target)

  def _train_on_full_data(self, data, target):
    prediction = self.forward(data)
    residuals = self.loss(prediction, target)
    residuals.backward()
    self.optimizer.step()
    return residuals

  def _train_on_batch(self, data, target, batch_size):
    max_size = data.size()[0]
    for start_index in range(0, max_size, batch_size):
      end_index = start_index + batch_size
      residuals = self._train_on_full_data(data[start_index:end_index], target[start_index:end_index])
    return residuals
        
  def freeze_layers(self, level):
    for layer in self.layers[:level]:
      if layer.type != 'Input': layer.freeze()   
  
  def _set_layer(self, layer_type, mark):
    for layer in self.layers:
      if layer.type == layer_type: layer.is_on = mark