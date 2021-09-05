import argparse
import os
import shutil
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
        sys.stdout.flush()
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    #validate(test_loader, model, criterion, normalizer, test=True)
    model.fc_out = Identity() 
    if args.cuda:
        model.cuda()
    extract(test_loader, model, criterion, normalizer, test=True)

def extract(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    features = []
    labels = []
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if model_args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        features.extend(output.cpu().detach().squeeze().numpy())
        labels.extend(batch_cif_ids)
        print(str(i)+' / '+str(len(val_loader)))
        sys.stdout.flush()
    features = np.array(features)
    pd.DataFrame(features, index=labels).to_csv('cgcnn_extracted.csv')
    print('saved')
    sys.stdout.flush()

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
