# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import csv
import os
import numpy as np
import pandas as pd

from domainbed.settings import SELECTION_METHODS, get_dataset_info, OOD_DETECTION_METHODS, EVAL_METRICS
from statistics import mean, stdev


def load_ood_performance(n_envs, ood_classes, selection_methods, input_dir, prefix, input_foldername):
    metrics = EVAL_METRICS
    results = {}
    for metric in metrics:
        results[metric] = {}
        for selection_method in selection_methods:
            results[metric][selection_method.name] = {}
            for test_env in range(n_envs):
                results[metric][selection_method.name][test_env] = {}
                for ood_class in ood_classes:
                    results[metric][selection_method.name][test_env][ood_class] = {}
                    result_filename = os.path.join(input_dir, f'{prefix}_{ood_class}', input_foldername,
                                               selection_method.name, metric, f'test_{test_env}_ood_{ood_class}_ood_detection_results.csv')

                    df = pd.read_csv(result_filename)
                    df = df.rename(columns={'Unnamed: 0': 'Method'})

                    # Iterate through the rows and populate the dictionaries
                    for index, row in df.iterrows():
                        method = row['Method']
                        results[metric][selection_method.name][test_env][ood_class][method] = row['Avg']
    return results


def summarize_ood_performance(n_envs, ood_classes, selection_methods, ood_methods, input_dir, prefix, all_results, finetune=True):
    for metric, results in all_results.items():
        for selection_method in selection_methods:
            mean_values = {}
            se_values = {}
            if finetune:
                summary_dir = os.path.join(input_dir, f'meta_ood_summary', selection_method.name, metric)
            else:
                summary_dir = os.path.join(input_dir, f'ood_summary', selection_method.name, metric)
            os.makedirs(summary_dir, exist_ok=True)
            for test_env in range(n_envs):
                mean_values[test_env] = []
                se_values[test_env] = []
                summary = []
                for method in ood_methods:
                    values = [float(v) for ood_class in ood_classes for k, v in results[selection_method.name][test_env][ood_class].items() if k == method]
                    mean_values[test_env].append(np.around(mean(values), decimals=2))
                    se_values[test_env].append(np.around(stdev(values)/np.sqrt(len(values)), decimals=2))
                    if len(values) > 1:
                        summary.append([method, f"{np.around(mean(values), decimals=2)} +/- {np.around(stdev(values)/np.sqrt(len(values)), decimals=2)}"])
                    else:
                        summary.append([method, np.around(mean(values), decimals=2)])

                # per test env summary
                result_filename = os.path.join(summary_dir, f'test_{test_env}.csv')
                header = ['Method', metric]
                with open(result_filename, 'w') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(header)
                    csvwriter.writerows(summary)

            # full summary
            result_filename = os.path.join(summary_dir, f'{metric}_full_summary.csv')
            header = ['Method'] + [f"test env {i}" for i in range(n_envs)] + [metric]
            full_summary = [header]

            # get the avg performance across OOD settings for each OOD detection method
            for i, method in enumerate(ood_methods):
                mean_list = [mean_values[test_env][i] for test_env in range(n_envs)]
                std_list = [se_values[test_env][i] for test_env in range(n_envs)]
                value_list = [f"{mean_list[i]} +/- {std_list[i]}" for i in range(n_envs)]
                row = [method] + value_list + [f"{np.around(mean(mean_list), decimals=2)} +/- {np.around(stdev(mean_list)/np.sqrt(len(mean_list)), decimals=2)}"]
                full_summary.append(row)

            # get the avg mean AUROC across OOD detection methods for each test env
            row = []
            for j in range(n_envs):
                values = [float(full_summary[i+1][j+1].split(" +/- ")[0]) for i in range(len(ood_methods))]
                row += [np.around(mean(values), decimals=2)]
            row = ['Avg'] + row + [f"{np.around(mean(row), decimals=2)} +/- {np.around(stdev(row)/np.sqrt(len(row)), decimals=2)}"]
            full_summary.append(row)

            # Transpose the full summary
            full_summary = np.array(full_summary).T

            with open(result_filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(full_summary)
            print(f"\n{selection_method.name} {metric} summary:")
            for row in full_summary:
                print(*row, sep='\t')


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ood_classes', type=int, nargs='+', default=None)
    parser.add_argument('--finetune', type=bool, default=False)
    args = parser.parse_args()

    input_dir = args.input_dir
    prefix = args.prefix
    dataset = args.dataset
    ood_classes = args.ood_classes
    finetune = args.finetune
    ood_methods = OOD_DETECTION_METHODS

    num_classes, n_envs = get_dataset_info(dataset)

    input_foldername = 'OOD_Detection_Results'
    results = load_ood_performance(n_envs, ood_classes, SELECTION_METHODS, input_dir, prefix, input_foldername)

    summarize_ood_performance(n_envs, ood_classes, SELECTION_METHODS, ood_methods, input_dir, prefix, results, finetune=finetune)
