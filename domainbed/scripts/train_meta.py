# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import csv
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torch
import copy
import torchvision
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import defaultdict
from argparse import Namespace
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.misc import get_the_best_models_configurations
from domainbed import model_selection
from domainbed.scripts.ood_detection import perform_ood_detection
from domainbed.settings import get_dataset_info, OOD_DETECTION_METHODS


def save_features(algorithm, input_dir, args, hparams):
    args = Namespace(**args)
    algorithm_dict = algorithm.state_dict()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']
        if args.ood_classes:
            if args.dataset in SMALL_IMAGES:
                dataset = vars(datasets)[f'{args.dataset}OOD'](args.data_dir, args.test_envs, hparams, args.ood_classes,
                                                               exclude_oods_from_training=True,
                                                               exclude_oods_from_testing=False)
            else:
                dataset = vars(datasets)[f'{args.dataset}OOD'](args.data_dir, args.test_envs, hparams, args.ood_classes,
                                                               no_data_aug=True,
                                                               exclude_oods_from_training=True,
                                                               exclude_oods_from_testing=False)
        else:
            raise Exception(f"OOD classes are not specified!")
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # Finite data loaders
    train_loaders = [DataLoader(env, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
                     for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs]

    test_loaders = [DataLoader(env, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS)
                    for i, (env, env_weights) in enumerate(in_splits) if i in args.test_envs]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # save features (featurizer outputs, logits, and labels)
    with torch.no_grad():
        train_output_list = [[] for _ in range(3)]
        train_file_name_list = ['ind_train_features', 'ind_train_logits', 'ind_train_labels']

        output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)

        for data_loader in train_loaders:
            for x, y in data_loader:
                x = x.to(device)
                y = y.detach().cpu().numpy()
                features = algorithm.featurizer(x).detach().cpu().numpy()
                logits = algorithm.predict(x).detach().cpu().numpy()

                for i, outputs in enumerate([features, logits, y]):  # order should match the name list
                    train_output_list[i].append(outputs.copy())

        for i, l in enumerate(train_output_list):
            np.save(os.path.join(output_dir, train_file_name_list[i]), np.concatenate(l))

        test_output_list = [[] for _ in range(6)]
        test_file_name_list = ['ind_test_features', 'ood_test_features',
                               'ind_test_logits', 'ood_test_logits',
                               'ind_test_labels', 'ood_test_labels']

        for data_loader in test_loaders:
            for x, y in data_loader:
                ood_mask = sum(y == i for i in args.ood_classes).bool().numpy()
                ind_mask = ~ood_mask

                x = x.to(device)
                y = y.cpu().numpy()
                features = algorithm.featurizer(x).cpu().numpy()
                logits = algorithm.predict(x).cpu().numpy()

                i = 0
                for outputs in [features, logits, y]:  # order should match the name list
                    for mask in [ind_mask, ood_mask]:  # order should match the name list
                        test_output_list[i].append(outputs[mask].copy())
                        i +=1

        for i, l in enumerate(test_output_list):
            np.save(os.path.join(output_dir, test_file_name_list[i]), np.concatenate(l))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="MADOD")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=3000,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--adapt_steps', type=int, default=5,
                        help='Number of adapt steps.')
    parser.add_argument('--checkpoint_freq', type=int, default=500,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--ood_classes', type=int, nargs='+', default=None)
    parser.add_argument('--n_tasks', type=int, default=4)
    parser.add_argument('--n_pseudo', type=int, default=1)
    parser.add_argument('--k_support', type=int, default=5)
    parser.add_argument('--k_query', type=int, default=5)
    parser.add_argument('--k_pseudo', type=int, default=5)
    parser.add_argument('--k_adapt', type=int, default=5)
    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--use_checkpoint_lr', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--summary_dir', type=str, default="summary")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0

    if args.load_checkpoint:
        if len(args.test_envs) > 1:
            raise NotImplementedError("Can only works with one test env.")
        else:
            test_env = args.test_envs[0]

        if len(args.ood_classes) > 1:
            raise NotImplementedError("Can only works with one ood class.")
        else:
            ood_class = args.ood_classes[0]

        selection_method = model_selection.LeaveOneOutSelectionMethod  # only works with leave one out

        best_models_configs = get_the_best_models_configurations(os.path.join(args.checkpoint_dir, f"ood_{ood_class}"),
                                                                 'MADOD',  # load MADOD checkpoints
                                                                 args.dataset, test_env)

        target_configs = best_models_configs[selection_method.name]

        # find the best config across all trials
        best_trial = 0
        best_acc = 0.
        for trial, config in target_configs.items():
            if config['test_acc'] > best_acc:
                best_trial = trial
                best_acc = config['test_acc']
        best_config = target_configs[best_trial]

        args.output_dir = os.path.join(best_config['args']['output_dir'], 'meta')  # update the output_dir

        best_model = torch.load(os.path.join(best_config['args']['output_dir'], f"model_step{best_config['step']}.pkl"))
        algorithm_dict = best_model['model_dict']
    else:
        algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.test_envs)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), args.test_envs)
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # use checkpoint hyperparameters
    if args.load_checkpoint and args.use_checkpoint_lr:
        hparams['lr'] = best_model['model_hparams']['lr']
        hparams['inner_lr'] = best_model['model_hparams']['inner_lr']
        hparams['weight_decay'] = best_model['model_hparams']['weight_decay']

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        if args.ood_classes:
            print("Running with OOD setting")
            dataset = vars(datasets)[f'{args.dataset}OOD'](args.data_dir, args.test_envs, hparams, args.ood_classes)
            dataset_meta = vars(datasets)[f'{args.dataset}MetaOOD'](args.data_dir, args.test_envs, hparams,
                                                                    args.ood_classes,
                                                                    n_pseudo=args.n_pseudo,
                                                                    k_support=args.k_support,
                                                                    k_query=args.k_query,
                                                                    k_pseudo=args.k_pseudo,
                                                                    use_relative_labels=False)
            dataset_adapt = vars(datasets)[f'{args.dataset}MetaOOD'](args.data_dir, args.test_envs, hparams,
                                                                     args.ood_classes,
                                                                     n_pseudo=0,
                                                                     k_support=args.k_adapt,
                                                                     k_query=args.k_adapt,
                                                                     k_pseudo=0,
                                                                     use_relative_labels=False)
        else:
            dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selection method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    meta_train_loaders = [DataLoader(env, batch_size=1, shuffle=False)
                          for env_i, env in enumerate(dataset_meta)
                          if env_i not in args.test_envs]

    meta_adapt_loaders = [DataLoader(env, batch_size=1, shuffle=False)
                          for env_i, env in enumerate(dataset_adapt)
                          if env_i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # Perform OOD detection
    save_features(algorithm, args.output_dir, vars(args), hparams)
    fpr95tpr_dict, auroc_dict, aupr_dict = perform_ood_detection(args.output_dir, args.ood_classes[0], get_dataset_info(args.dataset)[0])

    print('Initial FPR95TPR:')
    print(fpr95tpr_dict)
    print(np.mean(list(fpr95tpr_dict.values())))
    print()
    print('Initial AUROC:')
    print(auroc_dict)
    print(np.mean(list(auroc_dict.values())))
    print()
    print('Initial AUPR:')
    print(aupr_dict)
    print(np.mean(list(aupr_dict.values())))

    meta_train_minibatches_iterator = zip(*meta_train_loaders)

    meta_adapt_minibatches_iterator = zip(*meta_adapt_loaders)

    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    def create_task(minibatches):
        all_x_spt, all_y_spt, all_x_qry, all_y_qry, all_x_psd = [
            torch.cat([tensor.squeeze(0) for tensor in tensors])
            for tensors in zip(*minibatches)
        ]

        def balanced_sampling(x, y):
            # If empty directly return
            if len(x) == 0:
                return x, y

            fraction = 1 / len(minibatches)

            # Group indices by class
            class_indices = defaultdict(list)
            for idx, label in enumerate(y):
                class_indices[label.item()].append(idx)

            # Calculate number of samples per class
            n_per_class = min(len(indices) for indices in class_indices.values())
            n_keep_per_class = int(n_per_class * fraction)

            # Randomly sample from each class
            sampled_indices = []
            for indices in class_indices.values():
                sampled_indices.extend(random.sample(indices, n_keep_per_class))

            # Use these indices to sample from x and y
            return x[sampled_indices], y[sampled_indices]

        # Apply balanced sampling to support and query sets
        sampled_x_spt, sampled_y_spt = balanced_sampling(all_x_spt, all_y_spt)
        sampled_x_qry, sampled_y_qry = balanced_sampling(all_x_qry, all_y_qry)

        n_psd_samples = len(all_x_psd)
        # If empty, do nothing
        if n_psd_samples == 0:
            sampled_x_psd = all_x_psd
        else:
            n_psd_keep = n_psd_samples // len(minibatches)
            psd_indices = random.sample(range(n_psd_samples), n_psd_keep)
            sampled_x_psd = all_x_psd[psd_indices]

        return sampled_x_spt, sampled_y_spt, sampled_x_qry, sampled_y_qry, sampled_x_psd

    # Initialize the SummaryWriter
    writer = SummaryWriter(args.summary_dir)

    # Create or open the CSV file for this adapt_steps value
    csv_path = os.path.join(args.output_dir, f'accuracies.csv')
    csv_file_exists = os.path.exists(csv_path)

    # Prepare the header
    header = ["Step"]
    for name in eval_loader_names:
        header.append(f'{name}_acc')
    header.extend(["Avg_In_Domain_Acc", "Test_Domain_Acc"])

    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header if file is new
        if not csv_file_exists:
            csv_writer.writerow(header)

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()

        tasks = []

        for t in range(args.n_tasks):
            minibatches_device = [(x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device), x_psd.to(device))
                                  for x_spt, y_spt, x_qry, y_qry, x_psd in next(meta_train_minibatches_iterator)]
            task = create_task(minibatches_device)
            tasks.append(task)

        step_vals = algorithm.update(tasks)

        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        checkpoint_freq = args.checkpoint_freq
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            # use an algorithm copy to do the evalation so that the parameters won't change
            algorithm_eval = copy.deepcopy(algorithm)

            for s in range(args.adapt_steps):
                minibatches_device = [(x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device), x_psd.to(device))
                                      for x_spt, y_spt, x_qry, y_qry, x_psd in next(meta_adapt_minibatches_iterator)]
                task = create_task(minibatches_device)

                step_vals = algorithm_eval.finetune(task)

            # Perform OOD detection
            save_features(algorithm_eval, args.output_dir, vars(args), hparams)
            fpr95tpr_dict, auroc_dict, aupr_dict = perform_ood_detection(args.output_dir, args.ood_classes[0], get_dataset_info(args.dataset)[0])

            # Log OOD detection results to TensorBoard
            for method in OOD_DETECTION_METHODS:
                writer.add_scalar(f'OOD_Detection/FPR95TPR/{method}', fpr95tpr_dict[method], step)
                writer.add_scalar(f'OOD_Detection/AUROC/{method}', auroc_dict[method], step)
                writer.add_scalar(f'OOD_Detection/AUPR/{method}', aupr_dict[method], step)

            # Calculate and log average metrics
            avg_fpr95tpr = np.mean(list(fpr95tpr_dict.values()))
            avg_auroc = np.mean(list(auroc_dict.values()))
            avg_aupr = np.mean(list(aupr_dict.values()))

            writer.add_scalar('OOD_Detection/FPR95TPR/Average', avg_fpr95tpr, step)
            writer.add_scalar('OOD_Detection/AUROC/Average', avg_auroc, step)
            writer.add_scalar('OOD_Detection/AUPR/Average', avg_aupr, step)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # Prepare a row for CSV
            csv_row = [step]
            in_domain_accs = []
            test_domain_acc = None
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm_eval, loader, weights, device)
                results[name+'_acc'] = acc
                csv_row.append(acc)

                # Check if this is an in-domain environment (not the test environment)
                env_num = int(name.split('_')[0][3:])  # Extract environment number
                if 'in' in name:
                    if env_num != args.test_envs[0]:
                        in_domain_accs.append(acc)
                    else:
                        test_domain_acc = acc

            # Calculate average in-domain accuracy
            avg_in_acc = np.mean(in_domain_accs) if in_domain_accs else None
            results['avg_in_acc'] = avg_in_acc

            # Add average in-domain and test domain accuracies to the CSV row
            csv_row.append(avg_in_acc)
            csv_row.append(test_domain_acc)

            # Write the row to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_row)

            # Add average OOD detection metrics to results
            results['avg_fpr95'] = avg_fpr95tpr
            results['avg_auroc'] = avg_auroc
            results['avg_aupr'] = avg_aupr

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            # Log results to TensorBoard
            for key in results_keys:
                writer.add_scalar(key, results[key], step)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    # Close the main SummaryWriter
    writer.close()
