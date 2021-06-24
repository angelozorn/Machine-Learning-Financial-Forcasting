import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TimeSeriesIterableDataset(IterableDataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.data = zip(self.x, self.y)

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for x, y in self.data:
            yield x, y

    def reset(self):
        return TimeSeriesIterableDataset(self.x, self.y)


class Sliding_Window_IterableDataset(object):
    def __init__(self, data, device):
        if device.type == "cuda":
            print("default_tensor_type=torch.cuda.FloatTensor")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.rows = data.shape[0]
        self.cols = data.shape[1]
        self.features = data.columns.values
        self.x = data.values[:, :-1]
        self.y = data['Label'].values.reshape(self.rows, 1)
        self.window_size = None

    def split(self, window_size, scaler=MinMaxScaler()):
        self.window_size = window_size
        train_out = []
        test_out = []

        for start_ind in range(self.rows - self.window_size - 1):
            end_ind = start_ind + self.window_size
            x_train = self.x[start_ind: end_ind, :]
            y_train = self.y[start_ind: end_ind, :]

            scaler.fit(x_train)
            x_train = scaler.transform(x_train)

            x_train = torch.tensor(x_train, requires_grad=False, device=self.device, dtype=torch.float32)
            y_train = torch.tensor(y_train, requires_grad=False, device=self.device, dtype=torch.float32)

            train_ds = TimeSeriesIterableDataset(x_train, y_train)
            train_out.append(train_ds)

            x_test = self.x[end_ind: end_ind + 1, :]
            y_test = self.y[end_ind: end_ind + 1, :]
            x_test = scaler.transform(x_test)

            x_test = torch.tensor(x_test, requires_grad=False, device=self.device, dtype=torch.float32)
            y_test = torch.tensor(y_test, requires_grad=False, device=self.device, dtype=torch.float32)

            test_ds = TimeSeriesIterableDataset(x_test, y_test)
            test_out.append(test_ds)

            window_data = zip(train_out, test_out)
            return window_data

class MLP(nn.Module):
    def __init__(self, input_neurons, hidden1_neurons, hidden2_neurons, output_neurons):
        super(MLP, self).__init__()
        self.h1 = torch.nn.Linear(input_neurons, hidden1_neurons)  # hidden layer 1
        self.h2 = torch.nn.Linear(hidden1_neurons, hidden2_neurons)  # hidden layer 2
        self.output = torch.nn.Linear(hidden2_neurons, output_neurons)  # output layer

    def forward(self, x):
        x = F.silu(self.h1(x))  # ReLU activation function for h1
        x = F.sigmoid(self.h2(x))  # Sigmoid activation function for h2
        x = self.output(x)  # linear output
        return x
def Loss():
    def neg_log_likelihood(y_true, y_pred):
        return -(1/y_true.size(0))*torch.sum((y_true*torch.log(y_pred) + (1-y_true)*(torch.log(1-y_pred))))
    return neg_log_likelihood

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=9e-3, verbose=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.last = 1e5
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = min_delta

    def __call__(self, val_loss, model):
        score = val_loss
        diff = abs(score - self.last)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score and diff > self.min_delta:
            self.counter += 1
            if self.verbose >= 1:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        self.last = score

    def save_checkpoint(self, val_loss, model):
        if self.verbose >= 2:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
