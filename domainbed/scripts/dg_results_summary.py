# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import csv
import os
import numpy as np

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import reporting
from domainbed.settings import SELECTION_METHODS
from domainbed.lib.query import Q
from domainbed.scripts.collect_results import format_mean


def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
                                                                 {**group, "sweep_acc": selection_method.sweep_acc(
                                                                     group["records"])}
                                                                 ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                 [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                              .filter_equals(
                    "dataset, algorithm, test_env",
                    (dataset, algorithm, test_env)
                ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]

        return list(col_labels), alg_names + table[0], means


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument('--ood_classes', type=int, nargs='+', default=None)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    prefix = args.prefix
    ood_classes = args.ood_classes

    for selection_method in SELECTION_METHODS:
        summary_dir = os.path.join(input_dir, f'dg_summary', selection_method.name)
        os.makedirs(summary_dir, exist_ok=True)
        summary = []
        mean_list = []
        for ood_class in ood_classes:
            records = reporting.load_records(os.path.join(input_dir, f'{prefix}_{ood_class}'))

            header, values, means = print_results_tables(records, selection_method, args.latex)
            summary.append([ood_class] + values)
            mean_list.append(means)

        mean_list = np.array(mean_list)
        mean_across_env = mean_list.mean(axis=0)
        mean_across_env_with_std_err = []
        for i in range(len(mean_across_env)):
            input_acc = mean_list[:, i] / 100
            _, _, mean_err = format_mean(input_acc, False)
            mean_across_env_with_std_err.append(mean_err)

        last_row = ['Avg'] + [values[0]] + mean_across_env_with_std_err + [
            "{:.1f} +/- {:.1f}".format(mean_across_env.mean(), mean_across_env.std() / np.sqrt(len(mean_across_env)))]
        summary.append(last_row)

        header = ['OOD Class'] + header
        result_filename = os.path.join(summary_dir, f'full_summary.csv')
        with open(result_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(summary)
