import retreivedata
import MLpytorch
import torch
import numpy as np

from torch.utils.data import IterableDataset, DataLoader
from torch.backends import cudnn


SEED = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device("cpu")
print('Device being used:', device)

df = retreivedata.ohlc_data('SPY')

window_size = 128
n_features = df.shape[1] - 1
model_params = {
    'input_neurons': n_features, # 5 input features: Open, High, Low, Close, Volume
    'hidden1_neurons': 4, # Reduce dimensionality to number that is a multiple of 2
    'hidden2_neurons': 2, # Reduce dimensionality further
    'output_neurons': 1
}
train_params = {
    'batch_size': 8,
    'epochs': 300,
    'learning_rate': 0.0000001,
    'verbose': 2
}
model_iterations = df.shape[0] - window_size - 1

window = MLpytorch.Sliding_Window_IterableDataset(df, device)

with torch.no_grad():
    model = MLpytorch.MLP(**model_params)
    model.cuda(device=device)
cudnn.benchmark = True

ind, hit_count, hit_ratio = 0, 0, 0
actual = torch.zeros(model_iterations, dtype=torch.float32)
predictions = torch.zeros(model_iterations, dtype=torch.float32)


def train(train_dataset, batch_size, epochs=500, learning_rate=0.001, verbose=1):
    iterations_per_epoch = round(window_size / batch_size)
    early_stopper = MLpytorch.EarlyStopping(patience=10, verbose=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = MLpytorch.Loss()
    for e in range(epochs):
        train = DataLoader(train_dataset.reset(), batch_size=batch_size, shuffle=False)
        running_loss = 0
        for i, (x_batch, y_batch) in enumerate(train):
            y_pred = model.forward(x_batch)
            loss = loss_fn(y_batch, y_pred)
            #with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            if verbose >= 3:
                print('    Iteration {}/{}:'.format(i + 1, iterations_per_epoch), loss)
        epoch_loss = running_loss / iterations_per_epoch
        if verbose >= 2:
            print('  Epoch #{}'.format(e), ' --->  Loss: {}'.format(epoch_loss))
        early_stopper(epoch_loss, model)
        if early_stopper.early_stop:
            break
    if verbose >= 1:
        if early_stopper.early_stop:
            print('{}/{} Early Stopping at Epoch #{}'.format(ind + 1, model_iterations, e), ' --->  Best Loss: {}'.format(early_stopper.best_score))
        else:
            print('{}/{} Final Epoch #{}'.format(ind + 1, model_iterations, e), ' --->  Final Loss: {}'.format(epoch_loss))


for trainSet, testSet in window.split(window_size=window_size):
    train(train_dataset=trainSet, **train_params)
    model.load_state_dict(torch.load('checkpoint.pt'))

    ind += 1
    print()