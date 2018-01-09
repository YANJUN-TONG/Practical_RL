import os
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    def to_var(x, requires_grad=False, gpu=None):
        x = x.cuda(gpu)
        return Variable(x, requires_grad=requires_grad)
else:
    def to_var(x, requires_grad=False, vgpu=None):
        return Variable(x, requires_grad=requires_grad)


class EnhancedWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logdir = list(self.all_writers.keys())[0]

    def in_logdir(self, path):
        return os.path.join(self.logdir, path)

    def save(self, model, path):
        torch.save(model.state_dict(), self.in_logdir(path))

    def export_logs(self, filename='training.json'):
        self.export_scalars_to_json(self.in_logdir(filename))


def clip_grads(net, low=-10, high=10):
    """Gradient clipping to the range [low, high]."""
    for p in net.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(low, high)


def total_weights(net):
    '''Count total weights size.'''
    ret = 0
    for p in net.parameters():
        ret += p.data.cpu().numpy().size
    return ret
