import numpy as np
import os
import argparse
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Bernoulli
from matplotlib import pyplot as plt
import inspect
import ipdb

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
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type = int, default = 5000,
                    help = 'how many iterations to wait before logging training status')
parser.add_argument('--K', type = int, default = 10,
                    help = 'number of classes')
parser.add_argument('--N', type = int, default = 30,
                    help = 'number of categorical distributions')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


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
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.N = N
        self.temperature = tau


    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits):
        sample = Variable(self._sample_gumbel(logits.size()[-1]))
        if logits.is_cuda:
            sample = sample.cuda()
        y = logits + sample
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
        return q_y, logits_y

    def _decoder(self, y):
        h3 = self.relu(self.fc4(y))
        h4 = self.relu(self.fc5(h3))
        logits_x = self.fc6(h4)
        p_x = self.sigmoid(logits_x)
        return p_x, logits_x

    def forward(self, x):
        q_y, logits_y = self._encoder(x.view(-1, 784))
        y = self._gumbel_softmax(logits_y).view(-1, self.N * self.K)
        p_x, logits = self._decoder(y)
        return p_x, q_y, logits

def loss_fn(q_y, p_x, x, N, K, logits):
    prior = Variable(torch.log(torch.from_numpy(np.array([1.0/K]))).type(torch.FloatTensor)).cuda()
    kl = (q_y * (torch.log(q_y+1e-20) - prior)).view(-1, N, K)
    KL = torch.sum(kl)
    data = x.view(-1, 784)
    bceloss = nn.BCELoss(size_average=False)
    elbo = -bceloss(p_x, data) - KL
    loss = -(1.0/p_x.size()[0])*elbo
    return loss

def train(model, optimizer, train_loader, num_iters, dat, args):
    model.train()
    train_loss = 0
    np_temp = model.temperature
    for i, (data, _) in enumerate(train_loader):
        num_iters += 1
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        p_x, q_y, logits = model(data)
        loss = loss_fn(q_y, p_x, data, args.N, args.K, logits)
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

def test(model, test_loader):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        #p_x, q_y, logits = model(data)
        #test_loss += loss_fn(q_y, p_x, data, N, K).data[0]

def img_gen(model, args):
    model.eval()
    M = 100 * args.N
    np_y = np.zeros((M, args.K))
    np_y[range(M), np.random.choice(args.K,M)] = 1
    y = Variable(torch.from_numpy(np.reshape(np_y, [100, args.N*args.K])).type(torch.FloatTensor))
    if args.cuda:
        y = y.cuda()


    p_x, logits_x = model._decoder(y)

    np_y = np_y.reshape((10, 10, args.N, args.K))
    np_y = np.concatenate(np.split(np_y,10,axis=0),axis=3)
    np_y = np.concatenate(np.split(np_y,10,axis=1),axis=2)
    y_img = np.squeeze(np_y)

    x_p = p_x.view(10, 10, 28, 28).cpu().data.numpy()
    x_p = np.concatenate(np.split(x_p, 10, axis=0), axis=3)
    x_p = np.concatenate(np.split(x_p, 10, axis=1), axis=2)
    x_img = np.squeeze(x_p)
    f,axarr=plt.subplots(1, 2, figsize=(15,15))
    # samples
    axarr[0].matshow(y_img, cmap=plt.cm.gray)
    axarr[0].set_title('Z Samples')
    # reconstruction
    axarr[1].imshow(x_img, cmap=plt.cm.gray, interpolation='none')
    axarr[1].set_title('Generated Images')

    f.tight_layout()
    f.savefig('code_torch.png')

def main(args):
    torch.manual_seed(100)
    if args.cuda:
        torch.cuda.manual_seed(100)

    data_path = os.path.expanduser('~/data/mnist')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train = True, download = True,
                   transform = transforms.ToTensor()),
        batch_size = args.batch_size, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train = False, transform = transforms.ToTensor()),
        batch_size = args.batch_size, shuffle = True, **kwargs)

    num_iters = 0
    dat = []
    model = CVAE(args.N, args.K, args.tau)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    while num_iters <= args.num_iters:
        model, optimizer, num_iters, dat = train(model,
            optimizer, train_loader, num_iters, dat, args)
    #test(model, test_loader)
    img_gen(model, args)
    dat=np.array(dat).T
    f,axarr=plt.subplots(1,2)
    axarr[0].plot(dat[0],dat[1])
    axarr[0].set_ylabel('Temperature')
    axarr[1].plot(dat[0],dat[2])
    axarr[1].set_ylabel('-ELBO')
    plt.show()



if __name__ == '__main__':
    with ipdb.slaunch_ipdb_on_exception():
        main(args)
