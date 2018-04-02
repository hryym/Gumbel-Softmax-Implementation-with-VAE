import numpy as np
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli

parser = argparse.ArgumentParser(description = "categorical VAE with MNIST")
parser.add_argument('--batch_size', type = int, default = 100,
                    help = 'input batch size for training (default: 100)')
parser.add_argument('--num_iters', type = int, default = 50000,
                    help = 'number of iterations for training (default: 50000)')
parser.add_argument('--lr', type = float, default = 0.001,
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--tau', type = float, default = 1,
                    help = 'initial temperature for gumbel softmax (default: 1)')
parser.add_argument('--anneal_rate', type = float, default = 0.00003,
                    help = 'anneal rate for temperature (default: 0.00003)')
parser.add_argument('--min_temp', type = float, default = 0.5,
                    help = 'minimum temperature (default: 0.5)')
parser.add_argument('--use_cuda', action = 'store_true', default = 'False',
                    help = 'enables CUDA training (default = False)')
parser.add_argument('--log_interval', type = int, default = 5000,
                    help = 'how many iterations to wait before logging training status')
parser.add_argument('--K', type = int, default = 10,
                    help = 'number of classes')
parser.add_argument('--N', type = int, default = 30,
                    help = 'number of categorical distributions')
args = parser.parse_args()
args.cuda = torch.cuda.is_available() and args.use_cuda

data_path = os.path.expanduser('~/data/mnist')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train = True, download = True,
                   transform = transforms.ToTensor()),
    batch_size = args.batch_size, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train = False, transform = transforms.ToTensor()),
    batch_size = args.batch_size, shuffle = True, **kwargs)

class CVAE(nn.Module):
    def __init__(self, N, K, tau):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, N*K)
        self.fc4 = nn.Linear(N*K, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.K = K
        self.N = N
        self.temperature = tau


    def _sample_gumbel(self, shape, eps=1e-20):
        print('_sample_gumbel shape', shape)
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits):
        y = logits + self._sample_gumbel(list(logits.size()[-1]))
        return self.softmax(y / self.temperature)

    def _gumbel_softmax(self, logits, hard=False):
        y = self._gumbel_softmax_sample(logits)
        return y

    def _encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        logits_y = h3.view(-1, self.K)
        q_y = self.softmax(logits_y)
        return q_y

    def _decoder(self, y):
        h3 = self.relu(self.fc4(y))
        h4 = self.relu(self.fc5(h3))
        logits_x = self.fc6(h4)
        p_x = Bernoulli(logits=logits_x)
        return p_x

    def forward(self, x):
        q_y = self._encoder(x)
        y = torch.self._gumbel_softmax(q_y).view(-1, self.N * self.K)
        p_x = _decoder(y)
        return p_x, q_y

def loss_fn(q_y, p_x, x, N, K):
    kl = (q_y * (torch.log(q_y) - torch.log(1.0/K))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl, 2), 1)
    elbo = torch.sum(p_x.log_prob(x), 1) - KL
    loss = torch.mean(-elbo)
    return loss

def train(model, optimizer, num_iters, dat, args):
    model.train()
    train_loss = 0
    for i, (data, _) in enumerate(train_loader):
        num_iters += 1
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        p_x, q_y = model(data)
        loss = loss_fn(q_y, p_x, data, N, K)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if num_iters % 100 == 1:
            dat.append([num_iters, np_temp, loss.data[0]])
        if num_iters % 1000 == 1:
            np_temp = np.maximum(args.tau * np.exp(-args.anneal_rate * num_iters),
                                args.min_temp)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            model.temperature = np_temp
        if num_iters % args.log_interval == 1:
            print('Step %d, ELBO: %0.3f' % (num_iters,-loss.data[0]))
    return model, optimizer, num_iters, dat

    def test(model):
        model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            p_x, q_y = model(data)
            test_loss += loss_fn(q_y, p_x, data, N, K).data[0]

def main(args):
    num_iters = 0
    dat = []
    model = CVAE(args.N, args.K, args.tau)
    if args.cuda():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    while num_iters <= args.num_iters:
        model, optimizer, num_iters, dat = train(model,
            optimizer, num_iters, dat, args)
        test(model)
