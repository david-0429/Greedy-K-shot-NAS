import os
import sys
import torch
import argparse
import torch.nn as nn
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
from datetime import datetime
import logging
import argparse
import wandb
import pdb
import matplotlib.pyplot as plt
from Supernet.network.Kshot_network import KshotModel, SimplexNet
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters, to_onehot
from flops import get_cand_flops


class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--total-iters', type=int, default=150000, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.12, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='report frequency')
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=10000, help='epchos')
    parser.add_argument('--save-interval', type=int, default=1000, help='report frequency')
    
    #K-shot
    parser.add_argument('--k', type=int, default=2, help='k-supernet')
    parser.add_argument('--m', type=int, default=16, help='m paths')
    parser.add_argument('--warm-up', type=int, default=5, help='warm-up epochs')

    parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')
    parser.add_argument('--data-num', type=int, default=1280000, help='Number of train Data: Imagenet-100 = 1280000')
    

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    #wandb init
    print("wandb init")
    def get_timestamp():
        return datetime.now().strftime("%b%d_%H-%M-%S")
  
    wandb.init(
      # Set the project where this run will be logged
      project="Greedy K-shot NAS superent", 
      name=f"CIFAR10_{args.batch_size}_{args.k}_{args.m}_{args.learning_rate}_{args.total_iters}_{get_timestamp()}",
      config = {
        "dataset": "ImageNet-100",
        "Augmentation": "flip, ColorJitter",
        "batch_size": args.batch_size,
        "total_iters": args.total_iters,
        "loss_fn": "CrossEntropyLabelSmooth",
        "label-smooth": args.label_smooth,
        "optimizer": "SGD",
        "lr_name": "CosineAnnealingLR", 
        "init_lr": args.learning_rate, 
        "momentum": args.momentum, 
        "weight_decay": args.weight_decay, 
        "k": args.k,
        "m": args.m,
        "warm_up_epochs": args.warm_up, 
        "use_gpu": use_gpu
        }
      )
    #wandb.watch(model, loss_function, log="all")

    assert os.path.exists(args.train_dir)
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)

    assert os.path.exists(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            OpencvResize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])),
        batch_size=200, shuffle=False,
        num_workers=1, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    model = KshotModel(args.k)
    simplex_net = SimplexNet(4, args.k)

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    simpex_optimizer = torch.optim.SGD(get_parameters(simplex_net),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    criterion_smooth = CrossEntropyLabelSmooth(100, 0.1)

    if use_gpu:
        #model = nn.DataParallel(model)
        #simplex_net = nn.DataParallel(simplex_net)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=args.total_iters, last_epoch=-1)
    simpex_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(simpex_optimizer,
                    T_max=args.total_iters, last_epoch=-1)

    model = model.to(device)
    simplex_net = simplex_net.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            #simplex_net.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    args.optimizer = optimizer
    args.simpex_optimizer = simpex_optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.simpex_scheduler = simpex_scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint, strict=True)
            #simplex_net.load_state_dict(checkpoint, strict=True)
            validate(model, simplex_net, device, args, all_iters=all_iters)
        exit(0)

    while all_iters < args.total_iters:
        all_iters = train(model, simplex_net, device, args, val_interval=args.data_num//args.batch_size, bn_process=False, all_iters=all_iters)
    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

    #1 epoch : (args.count/args.batch_size)
    #1 epoch = val_interval
    wandb.finish()
    
    
def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, simplex_net, device, args, *, val_interval, bn_process=False, all_iters=None):

    optimizer = args.optimizer
    simpex_optimizer = args.simpex_optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    simpex_scheduler = args.simpex_scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    simplex_net.train()

    for iters in range(1, val_interval + 1):
        scheduler.step()
        simpex_scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()
        
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st


        get_random_cand = lambda:tuple(np.random.randint(12) for i in range(20))
        '''
        flops_l, flops_r, flops_step = 290, 360, 10
        bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]

        def get_uniform_sample_cand(*,timeout=500):
            idx = np.random.randint(len(bins))
            l, r = bins[idx]
            for i in range(timeout):
                cand = get_random_cand()
                if l*1e6 <= get_cand_flops(cand) <= r*1e6:
                    return cand
            return get_random_cand()
        '''
        def get_channel_factor():
            channel_fac = [0.2, 0.4, 0.6, 0.8, 1.0]
            idx = np.random.randint(len(channel_fac))
            return channel_fac[idx]

#----------------K-shot Supernet Train------------------

        if all_iters <= (args.data_num/args.batch_size)*args.warm_up or all_iters % 2 == 0:
            archi = (0, 1, 3, 2, 1, 0, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3, 2, 1, 2, 2) #get_random_cand()
            channel_fac = get_channel_factor()

            if all_iters <= (args.data_num/args.batch_size)*args.warm_up:
                lambdas = [[]]
                [lambdas[0].append(1/args.k) for i in range(args.k)]
            else:
                archi_encode = to_onehot(archi)
                archi_encode = archi_encode.to(device)
                lambdas = simplex_net(archi_encode)

            output = model(data, archi, channel_fac, lambdas)
            loss = loss_function(output, target)

            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            optimizer.step() #K-supernet trian
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            Top1_err += 1 - prec1.item() / 100
            Top5_err += 1 - prec5.item() / 100

            if all_iters % args.display_interval == 0:
                printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                            'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                            'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                            'lambda = {},\t'.format(lambdas[0]) + \
                            'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
                logging.info(printInfo)
                
                wandb.log({
                    "Top-1 train_err": Top1_err / args.display_interval,
                    "Top-5 train_err": Top5_err / args.display_interval,
                    "train_loss": loss.item(),
                    "lambdas": lambdas[0],
                    "lr": scheduler.get_lr()[0],
                    "train_iter": all_iters
                    })
                
                t1 = time.time()
                Top1_err, Top5_err = 0.0, 0.0
        
#----------------Simplex-net Train------------------
        else:
            data_chunk = torch.chunk(data, args.m, dim=0)
            output_list = []

            for i in range(args.m):
                archi = get_uniform_sample_cand()
                archi_encode = to_onehot(archi)
                archi_encode = archi_encode.to(device)
                lambdas = simplex_net(archi_encode)
                channel_fac = get_channel_factor()

                output_list.append(model(data_chunk[i], get_uniform_sample_cand(), channel_fac, lambdas))

            output = torch.cat(output_list, dim=0)
            loss = loss_function(output, target)

            simpex_optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            simpex_optimizer.step() #Simplex-net train
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            
            Top1_err += 1 - prec1.item() / 100
            Top5_err += 1 - prec5.item() / 100
            
            if all_iters % args.display_interval == 0:
                printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                            'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                            'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                            'lambda = {},\t'.format(lambdas[0]) + \
                            'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
                logging.info(printInfo)

                wandb.log({
                    "Top-1 train_err": Top1_err / args.display_interval,
                    "Top-5 train_err": Top5_err / args.display_interval,
                    "train_loss": loss.item(),
                    "lambdas": lambdas[0],
                    "lr": scheduler.get_lr()[0],
                    "train_iter": all_iters
                    })
                 
                t1 = time.time()
                Top1_err, Top5_err = 0.0, 0.0
        
#--------------------------------------------------------

        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, all_iters)

    return all_iters

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)

    wandb.log({
              "Top-1 val_err": 1 - top1.avg / 100,
              "Top-5 val_err": 1 - top5.avg / 100,
              "val_loss": objs.avg,
              "val_iter": all_iters
              })


if __name__ == "__main__":
    main()

