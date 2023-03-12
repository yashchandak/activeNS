import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, optim
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
# Learning rate decay

class BasicFit(nn.Module):
    def __init__(self):
        # super(BasicFit, self).__init__()
        super().__init__()


class Simple(BasicFit):
    def __init__(self, p):
        super(Simple, self).__init__()
        self.layer1 = nn.Linear(p, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = tensor(x).float()

        x = F.relu(self.layer1(x))
        return self.layer2(x)


class Linear(BasicFit):
    def __init__(self, p):
        # super(Linear, self).__init__()
        super().__init__()
        self.layer1 = nn.Linear(p, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = tensor(x).float()

        return self.layer1(x)


def fit(hyper, model, x, y, weights=None, batch_size=128):
    # opt = optim.Adam(model.parameters(), lr=3e-4)
    opt = optim.Adam(model.parameters(), lr=hyper['lr'], weight_decay=hyper['weight_decay'])
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    # opt = optim.SGD(self.parameters(), lr=1e-4)
    # opt = optim.RMSprop(self.parameters(), lr=1e-3)
    arr = []

    split=  hyper['split']
    ckpt = model.state_dict()
    best_val = 99999

    if weights is None:
        weights = np.ones(y.shape)

    N = x.shape[0]
    x, y, weights = tensor(x).float(), tensor(y).float(), tensor(weights).float()

    val_N = int(split * N)
    batch_size = N-val_N    # Full-batch GD

    if split > 0:
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        x_val, y_val =  x[idxs[:val_N]], y[idxs[:val_N]]       
        x_train, y_train =  x[idxs[val_N:]], y[idxs[val_N:]]
        w_val, w_train = weights[idxs[:val_N]], weights[idxs[val_N:]]
        # w_val_norm, w_train_norm = w_val, w_train
        w_val_norm, w_train_norm = w_val/w_val.sum(), w_train/w_train.sum()
    else:
        # Do not do any split for generlaization test
        # Just optimize on the training set as much as possible
        x_val, y_val =  x, y       
        x_train, y_train =  x, y
        w_val, w_train = weights, weights
        w_val_norm, w_train_norm = w_val/w_val.sum(), w_train/w_train.sum()


    ctr = 0
    while True:

        for i in range(int((N-val_N)/batch_size)):
            l, u = i*batch_size, i*batch_size + batch_size
            if l + batch_size > (N - val_N):                       
                u = N - val_N
                batch_size = u-l

            x_train_b = x_train[l:u]
            y_train_b = y_train[l:u]
            w_b = w_train_norm[l:u]

            opt.zero_grad()

            preds = model.forward(x_train_b)                         # Nx1
            train_loss = (preds - y_train_b)**2                           # Nx1
            # train_loss = torch.abs(preds - y_train)

            final_train_loss = torch.sum(train_loss*w_b) * (N-val_N)/batch_size           # mean(Nx1 * Nx1) -> 1
            final_train_loss.backward()
            opt.step()

        # scheduler.step()

        # Do early stopping
        preds = model.forward(x_val)                         # Nx1
        val_loss = (preds - y_val)**2    
        final_val_loss = torch.sum(val_loss*w_val_norm)           # mean(Nx1 * Nx1) -> 1
            
        arr.append(final_val_loss.item())
        # if not ctr%100: print(ctr, ' : ', final_train_loss.item(), final_val_loss.item())

        if final_val_loss.item() < best_val:
            best_val = final_val_loss.item()
            ckpt = model.state_dict()
            # ckpt = copy.deepcopy(model.state_dict())

        # Stopping criteria based on sliding window average improvement
        if len(arr) > 50 and np.mean(arr[-20:]) > (np.mean(arr[-30:-10]) - 1e-5):
            break

            scheduler.step()
            arr = []
    
        ctr += 1
        if ctr > 5000:
            break

    # plt.figure()
    # plt.scatter(w_val_norm.numpy(), val_loss.detach().numpy())
    # # plt.scatter(w_val_norm.numpy(), (preds-y_val).detach().numpy())
    # plt.show()


    model.load_state_dict(ckpt)