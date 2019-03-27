import torch

def make_holdout(data, labels, holdout_size = 0.3, train_size = 0.7, shuffle = True, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=holdout_size, shuffle=shuffle, random_state=random_state)
    X_train = torch.tensor(X_train, dtype=torch.float).cuda()
    X_test = torch.tensor(X_test, dtype=torch.float).cuda()
    y_train = torch.tensor(y_train, dtype=torch.float).view(len(y_train), 1).cuda()
    y_test = torch.tensor(y_test, dtype=torch.float).view(len(y_test), 1).cuda()
    return X_train, X_test, y_train, y_test
  
def to_cuda_tensor(data):
    return_data = torch.tensor(data, dtype=torch.float)
    if len(data.shape) == 1: return_data.view(len(data), 1)
    return return_data.cuda()