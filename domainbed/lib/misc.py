# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import os
import sys
import operator
import numpy as np
import torch

from numbers import Number
from collections import OrderedDict, Counter
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from domainbed.settings import SELECTION_METHODS
from domainbed.lib import reporting


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def accuracy_mixup(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def get_the_best_models_configurations(input_dir, algorithm, dataset, test_env):
    # check directory
    if not os.path.exists(input_dir):
        raise Exception(f"Folder {input_dir} does not exists!")

    records = reporting.load_records(input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r:
        r['dataset'] == dataset and
        r['algorithm'] == algorithm and
        r['test_env'] == test_env
    )

    best_models_configs = {}
    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')
        best_models_configs[selection_method.name] = {}

        for group in records:
            print(f"\ttrial_seed: {group['trial_seed']}")
            best_hparams = selection_method.hparams_accs(group['records'])

            best_acc, best_model_records = best_hparams[0]
            print(f"\t\tBest test_acc: {best_acc['test_acc']}")

            best_record_index = next((index for (index, d) in enumerate(best_model_records) if
                                      d[f"env{test_env}_in_acc"] == best_acc['test_acc']), None)
            best_record = best_model_records[best_record_index]
            best_record_dir = best_record['args']['output_dir']
            best_record_step = best_record['step']
            print(f"\t\tBest record dir: {best_record_dir}")
            print(f"\t\tBest record step: {best_record_step}")

            best_models_configs[selection_method.name][group['trial_seed']] = {'val_acc': best_acc['val_acc'],
                                                                               'test_acc': best_acc['test_acc'],
                                                                               'step': best_record_step,
                                                                               'args': best_record['args'],
                                                                               'hparams': best_record['hparams']}
    return best_models_configs


def get_fpr_at_95_tpr(tpr, fpr):
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def get_statistics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    fpr_at_95_tpr = get_fpr_at_95_tpr(tpr, fpr)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)

    # # reverse label
    # precision, recall, thresholds = precision_recall_curve(-(labels * 2 - 1), -scores)
    # aupr = auc(recall, precision)

    return fpr_at_95_tpr, auroc, aupr
