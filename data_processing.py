import torch
import numpy as np

def train_test_split(data, target, test_size=0.3, train_size=0.7, shuffle=True, random_state=None):
    train_len = int(data.shape[0] * train_size)
    np.random.seed(random_state)
    indexes = list(range(data.shape[0]))
    if shuffle: np.random.shuffle(indexes)
    return data[indexes[:train_len]], data[indexes[train_len:]], \
           target[indexes[:train_len]], target[indexes[train_len:]] 

def make_holdout(data, target, holdout_size = 0.3, train_size = 0.7, shuffle = True, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target, 
                                                        test_size=holdout_size,
                                                        shuffle=shuffle, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def to_cuda_tensor(data):
    if not torch.is_tensor(data): data = torch.tensor(data, dtype=torch.float)
    if len(data.shape) == 1: data = data.view(len(data), 1)
    return data.cuda()