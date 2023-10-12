import os
import re
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pdb


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,4)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, save="./experiment_0", tag=''):
    if not os.path.exists(save):
        os.makedirs("./model")
        os.makedirs("./model")
    filename = os.path.join(
        save + "/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)
    latestfilename = os.path.join(
        save + "/{}checkpoint-latest.pth.tar".format(tag))
    torch.save(state, latestfilename)


def get_lastest_model(save="./models/experiment_0"):
    if not os.path.exists(save):
        os.mkdir("./models/experiment_0")
    model_list = os.listdir(save)
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-2]
    iters = re.findall(r'\d+', lastest_model)
    return save + "/" + lastest_model, int(iters[0])


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


def to_onehot(a, c, a_choice, c_choice):
  a_one_hot = torch.nn.functional.one_hot(torch.tensor(a), num_classes=a_choice)
  c_one_hot = torch.nn.functional.one_hot(torch.tensor(c), num_classes=c_choice)

  a_one_hot = a_one_hot.to(torch.float32)
  c_one_hot = c_one_hot.to(torch.float32)
  return a_one_hot, c_one_hot


class PruningMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, n):
        self.n = n

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[self.n:] = 0
        return mask


def Pruned_model(module, name, n):
    PruningMethod.apply(module, name, n)
    prune.remove(module, name)
    return module

