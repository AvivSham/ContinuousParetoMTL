import random
import argparse
from pathlib import Path
from termcolor import colored

import numpy as np
import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms

from pareto.metrics import topk_accuracy
from pareto.optim import VisionHVPSolver, MINRESKKTSolver
from pareto.datasets import MultiMNIST
from pareto.networks import MultiLeNet
from pareto.utils import TopTrace

class LeNet(torch.nn.Module):
    def __init__(self, n_tasks):
        super(LeNet, self).__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)

        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if 'task' not in n])

    def forward(self, x, i=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))

        if i is not None:
            layer_i = getattr(self, 'task_{}'.format(i))
            return layer_i(x)

        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(x))

        return torch.stack(outs, dim=0)


class Dataset:

    def __init__(self, path, val_size=0.):
        self.path = path
        self.val_size = val_size

    def get_datasets(self):
        with open(self.path, 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

        n_train = len(trainX)
        if self.val_size > 0:
            trainX, valX, trainLabel, valLabel = train_test_split(
                trainX, trainLabel, test_size=self.val_size, random_state=42
            )
            n_train = len(trainX)
            n_val = len(valX)

        trainX = torch.from_numpy(trainX.reshape(n_train, 1, 36, 36)).float()
        trainLabel = torch.from_numpy(trainLabel).long()
        testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
        testLabel = torch.from_numpy(testLabel).long()

        train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
        test_set = torch.utils.data.TensorDataset(testX, testLabel)

        if self.val_size > 0:
            valX = torch.from_numpy(valX.reshape(n_val, 1, 36, 36)).float()
            valLabel = torch.from_numpy(valLabel).long()
            val_set = torch.utils.data.TensorDataset(valX, valLabel)

            return train_set, val_set, test_set

        return train_set, test_set


@torch.no_grad()
def evaluate(network, dataloader, device, closures, header=''):
    num_samples = 0
    losses = np.zeros(2)
    top1s = np.zeros(2)
    network.train(False)
    for images, labels in dataloader:
        batch_size = len(images)
        num_samples += batch_size
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        losses_batch = [c(network, logits, labels).item() for c in closures]
        losses += batch_size * np.array(losses_batch)
        top1s[0] += batch_size * topk_accuracy(logits[0], labels[:, 0], k=1)
        top1s[1] += batch_size * topk_accuracy(logits[1], labels[:, 1], k=1)
    losses /= num_samples
    top1s /= num_samples

    loss_msg = '[{}]'.format('/'.join([f'{loss:.6f}' for loss in losses]))
    top1_msg = '[{}]'.format('/'.join([f'{top1 * 100.0:.2f}%' for top1 in top1s]))
    msgs = [
        f'{header}:' if header else '',
        'loss', colored(loss_msg, 'yellow'),
        'top@1', colored(top1_msg, 'yellow')
    ]
    print(' '.join(msgs))
    return losses, top1s


def train(start_path, data_path, beta, lr, batch_size, num_steps):

    # prepare hyper-parameters

    seed = 42

    cuda_enabled = True
    cuda_deterministic = False

    num_workers = 2

    shared = False

    stochastic = False
    kkt_momentum = 0.0
    create_graph = False
    grad_correction = False
    shift = 0.0
    tol = 1e-5
    damping = 0.1
    maxiter = 50

    # lr = lr
    momentum = 0.0
    weight_decay = 0.0

    verbose = False


    # prepare path

    ckpt_name = start_path.name.split('.')[0]
    root_path = Path(__file__).resolve().parent
    dataset_path = root_path / 'MultiMNIST'
    ckpt_path = root_path / 'cpmtl' / ckpt_name

    if not start_path.is_file():
        raise RuntimeError('Pareto solutions not found.')

    root_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)


    # fix random seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_enabled and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # prepare device

    if cuda_enabled and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        if cuda_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
    else:
        device = torch.device('cpu')


    # prepare dataset

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #
    # trainset = MultiMNIST(dataset_path, train=True, download=True, transform=transform)
    #
    # testset = MultiMNIST(dataset_path, train=False, download=True, transform=transform)

    trainset, _, testset = Dataset(data_path, val_size=0.1).get_datasets()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # prepare network

    network = LeNet(n_tasks=2)
    network.to(device)


    # initialize network

    start_ckpt = torch.load(start_path, map_location='cpu')
    network.load_state_dict(start_ckpt['state_dict'])


    # prepare losses

    criterion = F.cross_entropy
    closures = [lambda n, l, t: criterion(l[0], t[:, 0]), lambda n, l, t: criterion(l[1], t[:, 1])]


    # prepare HVP solver

    hvp_solver = VisionHVPSolver(network, device, trainloader, closures, shared=shared)
    hvp_solver.set_grad(batch=False)
    hvp_solver.set_hess(batch=True)


    # prepare KKT solver

    kkt_solver = MINRESKKTSolver(
        network, hvp_solver, device,
        stochastic=stochastic, kkt_momentum=kkt_momentum, create_graph=create_graph,
        grad_correction=grad_correction, shift=shift, tol=tol, damping=damping, maxiter=maxiter)


    # prepare optimizer

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    # first evaluation

    losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}')


    # prepare utilities
    top_trace = TopTrace(len(closures))
    top_trace.print(tops, show=False)

    beta = beta.to(device)


    # training

    for step in range(1, num_steps + 1):

        network.train(True)
        optimizer.zero_grad()
        kkt_solver.backward(beta, verbose=verbose)
        optimizer.step()

        losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}: {step}/{num_steps}')

        top_trace.print(tops)

        ckpt = {
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'beta': beta,
        }
        record = {'losses': losses, 'tops': tops}
        ckpt['record'] = record
        torch.save(ckpt, ckpt_path / f'{step:d}.pth')

    hvp_solver.close()


def cpmtl(ckpt_path, data_path, lr, bs, n_steps):
    beta = torch.Tensor([1, 0])

    for start_path in sorted(Path(ckpt_path).glob('*.pth'), key=lambda x: int(x.name.split('.')[0])):
        train(start_path, data_path, beta, lr, bs, n_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="continuous PMTL MultiMNIST")
    parser.add_argument('--num-steps', type=int, default=10, help='num steps for expansion')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--data-path', type=str,
                        default='/cortex/data/images/paretoHN/data/multi_mnist.pickle', help='path for data')
    parser.add_argument('--ckpt-path', type=str,
                        default='/cortex/data/images/paretoHN/data/model', help='path for start solutions')

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    data_path = args.data_path
    lr = args.lr
    bs = args.batch_size
    niter = args.num_steps
    cpmtl(
        ckpt_path=ckpt_path,
        data_path=data_path,
        lr=lr,
        bs=bs,
        n_steps=niter
    )
