import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.float64


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray):
    return torch.from_numpy(ndarray).type(FLOAT)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def process_obs(observation):
    pass

def normal_action(action):
    pass

def reverse_action(action):
    pass

