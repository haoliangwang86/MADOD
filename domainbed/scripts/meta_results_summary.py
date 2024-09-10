# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import csv
import os
import numpy as np
import pandas as pd

from statistics import mean, stdev
from domainbed.settings import SELECTION_METHODS, get_dataset_info, OOD_DETECTION_METHODS, EVAL_METRICS
from domainbed.lib.misc import get_the_best_models_configurations
from domainbed.scripts.ood_results_summary import load_ood_performance, summarize_ood_performance


def load_test_domain_accuracy(n_envs, ood_classes, selection_methods, input_dir, prefix, input_foldername):
    metrics = ['Test_Domain_Acc']  # We only have one metric for test domain accuracy
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
                                                   selection_method.name, metric, f'test_{test_env}_ood_{ood_class}_test_domain_accuracy.csv')

                    if os.path.exists(result_filename):
                        df = pd.read_csv(result_filename)
                        df = df.rename(columns={'Unnamed: 0': 'Trial_Seed'})

                        # Iterate through the rows and populate the dictionaries
                        for index, row in df.iterrows():
                            trial_seed = row['Trial_Seed']
                            if trial_seed not in ['Avg', 'SE']:
                                results[metric][selection_method.name][test_env][ood_class][trial_seed] = row['Test_Domain_Acc']
                    else:
                        print(f"Warning: Result file not found: {result_filename}")
    return results


def summarize_test_domain_accuracy_performance(n_envs, ood_classes, selection_methods, input_dir, prefix, all_results):
    metric = 'Test_Domain_Acc'
    for selection_method in selection_methods:
        mean_values = {}
        se_values = {}
        summary_dir = os.path.join(input_dir, f'meta_dg_summary', selection_method.name)
        os.makedirs(summary_dir, exist_ok=True)

        for test_env in range(n_envs):
            mean_values[test_env] = []
            se_values[test_env] = []
            summary = []

            values = [float(v) for ood_class in ood_classes for v in
                      all_results[metric][selection_method.name][test_env][ood_class].values() if
                      not np.isnan(float(v))]

            if values:
                mean_value = np.around(np.mean(values), decimals=2)
                se_value = np.around(np.std(values, ddof=1) / np.sqrt(len(values)), decimals=2)
                mean_values[test_env].append(mean_value)
                se_values[test_env].append(se_value)

                if len(values) > 1:
                    summary.append([metric, f"{mean_value} +/- {se_value}"])
                else:
                    summary.append([metric, mean_value])
            else:
                mean_values[test_env].append(np.nan)
                se_values[test_env].append(np.nan)
                summary.append([metric, "N/A"])

            # per test env summary
            result_filename = os.path.join(summary_dir, f'test_{test_env}.csv')
            header = ['Metric', 'Value']
            with open(result_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)
                csvwriter.writerows(summary)

        # full summary
        result_filename = os.path.join(summary_dir, f'{metric}_full_summary.csv')
        header = ['Metric'] + [f"test env {i}" for i in range(n_envs)] + ['Average']
        full_summary = [header]

        # get the avg performance across OOD settings
        mean_list = [mean_values[test_env][0] for test_env in range(n_envs)]
        std_list = [se_values[test_env][0] for test_env in range(n_envs)]
        value_list = [f"{mean_list[i]} +/- {std_list[i]}" if not np.isnan(mean_list[i]) else "N/A" for i in
                      range(n_envs)]

        valid_means = [m for m in mean_list if not np.isnan(m)]
        if valid_means:
            overall_mean = np.around(np.mean(valid_means), decimals=2)
            overall_se = np.around(np.std(valid_means, ddof=1) / np.sqrt(len(valid_means)), decimals=2)
            overall_value = f"{overall_mean} +/- {overall_se}"
        else:
            overall_value = "N/A"

        row = [metric] + value_list + [overall_value]
        full_summary.append(row)

        # Transpose the full summary
        full_summary = np.array(full_summary).T

        with open(result_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(full_summary)

        print(f"\n{selection_method.name} {metric} summary:")
        for row in full_summary:
            print(*row, sep='\t')


def summarize_test_domain_accuracy(best_configs, input_filename, output_foldername, output_filename):
    test_domain_acc_dict = {}

    for trial_seed, config in best_configs.items():
        accuracy_file = os.path.join(config['args']['output_dir'], input_filename)

        if os.path.exists(accuracy_file):
            df = pd.read_csv(accuracy_file)
            test_domain_acc = df['Test_Domain_Acc'].iloc[-1]  # Get the last row's Test_Domain_Acc
            # Convert to percentage and round to 2 decimal places
            test_domain_acc_dict[trial_seed] = round(test_domain_acc * 100, 2)
        else:
            print(f"Warning: Accuracy file not found for trial_seed {trial_seed}. Using NA value.")
            test_domain_acc_dict[trial_seed] = None

    # Create DataFrame
    df = pd.DataFrame.from_dict(test_domain_acc_dict, orient='index', columns=['Test_Domain_Acc'])
    df.index.name = 'Trial_Seed'

    # Calculate average and SE, ignoring NA values
    valid_accuracies = df['Test_Domain_Acc'].dropna()
    if not valid_accuracies.empty:
        avg = np.mean(valid_accuracies)
        se = np.std(valid_accuracies, ddof=1) / np.sqrt(len(valid_accuracies))
    else:
        avg = np.nan
        se = np.nan

    # Add Avg and SE to the DataFrame
    df.loc['Avg'] = avg
    df.loc['SE'] = se

    # Round all numeric values to 2 decimal places
    df = df.round(2)

    # Save results
    os.makedirs(os.path.join(output_foldername, 'Test_Domain_Acc'), exist_ok=True)
    df.to_csv(os.path.join(output_foldername, 'Test_Domain_Acc', output_filename))

    print(f"Test Domain Accuracy results saved in {os.path.join(output_foldername, 'Test_Domain_Acc', output_filename)}")


def summarize_ood_performance_per_ood_per_test_env(best_configs, input_filename, output_foldername, output_filename):
    fpr95tpr_dict = {}
    auroc_dict = {}
    aupr_dict = {}
    methods = set()  # To keep track of all unique methods

    for trial_seed, config in best_configs.items():
        fpr95tpr_dict[trial_seed] = {}
        auroc_dict[trial_seed] = {}
        aupr_dict[trial_seed] = {}

        result_file = os.path.join(config['args']['output_dir'], input_filename)

        if os.path.exists(result_file):
            df = pd.read_csv(result_file)

            # Iterate through the rows and populate the dictionaries
            for index, row in df.iterrows():
                method = row['Method']
                methods.add(method)
                fpr95tpr_dict[trial_seed][method] = row['FPR95TPR']
                auroc_dict[trial_seed][method] = row['AUROC']
                aupr_dict[trial_seed][method] = row['AUPR']
        else:
            print(f"Warning: Result file not found for trial_seed {trial_seed}. Using NA values.")

    # Ensure all dictionaries have entries for all methods, filling with None if missing
    for trial_seed in best_configs.keys():
        for method in methods:
            fpr95tpr_dict[trial_seed].setdefault(method, None)
            auroc_dict[trial_seed].setdefault(method, None)
            aupr_dict[trial_seed].setdefault(method, None)

    # save results
    metrics = EVAL_METRICS
    for i, results in enumerate([fpr95tpr_dict, auroc_dict, aupr_dict]):
        df = pd.DataFrame(results)
        df = df.rename_axis('Method')

        # Calculate average and SE, ignoring NA values
        df['Avg'] = df.apply(lambda row: np.nanmean(row.dropna()), axis=1).round(2)
        df['SE'] = df.apply(lambda row: np.nanstd(row.dropna()) / np.sqrt(row.notna().sum()), axis=1).round(2)

        os.makedirs(os.path.join(output_foldername, metrics[i]), exist_ok=True)
        df.to_csv(os.path.join(output_foldername, metrics[i], output_filename))

    print(f"Results saved in {output_foldername}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ood_classes', type=int, nargs='+', default=None)
    parser.add_argument('--algorithm', default='MADOD')
    args = parser.parse_args()

    algorithm = args.algorithm
    input_dir = args.input_dir
    prefix = args.prefix
    dataset = args.dataset
    ood_classes = args.ood_classes
    ood_folder_name = 'OOD_Detection_Results'
    ood_methods = OOD_DETECTION_METHODS

    num_classes, n_envs = get_dataset_info(dataset)

    for ood_class in ood_classes:
        current_input_dir = os.path.join(input_dir, prefix+f'_{ood_class}')

        for test_env in range(n_envs):
            print(f"Test env: {test_env}")
            best_models_configs = get_the_best_models_configurations(current_input_dir, algorithm, dataset, test_env)

            # summarize per OOD class per test env OOD detection performance
            for selection_method, best_configs in best_models_configs.items():
                input_filename = os.path.join('meta', 'final_ood_detection_results.csv')
                output_foldername = os.path.join(current_input_dir, 'Meta_OOD_Detection_Results', selection_method)
                output_filename = f'test_{test_env}_ood_{ood_class}_ood_detection_results.csv'
                summarize_ood_performance_per_ood_per_test_env(best_configs, input_filename, output_foldername, output_filename)

                # Summarize Test Domain Accuracy
                test_acc_input_filename = os.path.join('meta', 'accuracies.csv')
                test_acc_output_foldername = os.path.join(current_input_dir, 'Meta_Test_Domain_Accuracy', selection_method)
                test_acc_output_filename = f'test_{test_env}_ood_{ood_class}_test_domain_accuracy.csv'
                summarize_test_domain_accuracy(best_configs, test_acc_input_filename, test_acc_output_foldername, test_acc_output_filename)

    input_foldername = 'Meta_OOD_Detection_Results'
    results = load_ood_performance(n_envs, ood_classes, SELECTION_METHODS, input_dir, prefix, input_foldername)
    summarize_ood_performance(n_envs, ood_classes, SELECTION_METHODS, ood_methods, input_dir, prefix, results)

    input_foldername = 'Meta_Test_Domain_Accuracy'
    test_domain_acc_results = load_test_domain_accuracy(n_envs, ood_classes, SELECTION_METHODS, input_dir, prefix, input_foldername)
    summarize_test_domain_accuracy_performance(n_envs, ood_classes, SELECTION_METHODS, input_dir, prefix, test_domain_acc_results)
