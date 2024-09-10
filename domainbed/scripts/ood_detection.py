# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from domainbed.lib.misc import get_statistics
from domainbed.scripts.gmm_utils import gmm_fit
from domainbed.settings import OOD_DETECTION_METHODS


def detect_oods(input_dir, method, ood_classes, num_classes):
    num_classes = num_classes - len(ood_classes)

    ind_logits = torch.from_numpy(np.load(os.path.join(input_dir, 'ind_test_logits.npy')))
    ood_logits = torch.from_numpy(np.load(os.path.join(input_dir, 'ood_test_logits.npy')))

    if method == 'energy':
        ind_scores = -torch.logsumexp(ind_logits, dim=1).cpu().numpy()
        ood_scores = -torch.logsumexp(ood_logits, dim=1).cpu().numpy()
    elif method == 'msp':
        ind_scores = -torch.max(torch.softmax(ind_logits, dim=1), dim=1)[0].cpu().numpy()
        ood_scores = -torch.max(torch.softmax(ood_logits, dim=1), dim=1)[0].cpu().numpy()
    elif method == 'ddu':
        ind_train_features = torch.from_numpy(np.load(os.path.join(input_dir, 'ind_train_features.npy')))
        labels = torch.from_numpy(np.load(os.path.join(input_dir, 'ind_train_labels.npy')))
        ind_test_features = torch.from_numpy(np.load(os.path.join(input_dir, 'ind_test_features.npy')))
        ood_features = torch.from_numpy(np.load(os.path.join(input_dir, 'ood_test_features.npy')))
        try:
            gaussians_model, jitter_eps = gmm_fit(embeddings=ind_train_features, labels=labels, num_classes=num_classes)
            ind_test_features = gaussians_model.log_prob(ind_test_features[:, None, :])
            ood_features = gaussians_model.log_prob(ood_features[:, None, :])
            ind_scores = -torch.logsumexp(ind_test_features, dim=1, keepdim=False).cpu().numpy()
            ood_scores = -torch.logsumexp(ood_features, dim=1, keepdim=False).cpu().numpy()
            ind_scores[np.isneginf(ind_scores)] = 0  # replace -inf with 0
            ood_scores[np.isneginf(ood_scores)] = 0  # replace -inf with 0
        except RuntimeError as e:
            print("Runtime Error caught: " + str(e))
    elif method == 'ocsvm':
        ind_train_features = np.load(os.path.join(input_dir, 'ind_train_features.npy'))
        ind_test_features = np.load(os.path.join(input_dir, 'ind_test_features.npy'))
        ood_features = np.load(os.path.join(input_dir, 'ood_test_features.npy'))

        ss = StandardScaler()
        ss.fit(ind_train_features)
        ind_train_features = ss.transform(ind_train_features)
        ind_test_features = ss.transform(ind_test_features)
        ood_features = ss.transform(ood_features)

        model = OneClassSVM(gamma=0.001, nu=0.001).fit(ind_train_features)
        ind_scores = -model.decision_function(ind_test_features)
        ood_scores = -model.decision_function(ood_features)
    else:
        print(f"{method} not implemented!")
        return

    scores = np.concatenate((ind_scores, ood_scores), axis=0)
    bin_labels = np.concatenate((np.zeros(ind_scores.shape[0]), np.ones(ood_scores.shape[0])), axis=0)

    fpr_at_95_tpr, auroc, aupr = get_statistics(bin_labels, scores)

    result_filename = os.path.join(input_dir, f"ood_detection_results.csv")
    header = ['Method', 'FPR95TPR', 'AUROC', 'AUPR']
    result = [method, np.around(100 * fpr_at_95_tpr, decimals=2), np.around(100 * auroc, decimals=2), np.around(100 * aupr, decimals=2)]

    if os.path.exists(result_filename):
        with open(result_filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(result)
    else:
        with open(result_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerow(result)

    return np.around(100 * fpr_at_95_tpr, decimals=2), np.around(100 * auroc, decimals=2), np.around(100 * aupr, decimals=2)


def perform_ood_detection(input_dir, ood_class, num_classes):
    results_dict = {method: {} for method in OOD_DETECTION_METHODS}

    for method in OOD_DETECTION_METHODS:
        fpr_at_95_tpr, auroc, aupr = detect_oods(input_dir, method, [ood_class], num_classes)
        results_dict[method]['FPR95TPR'] = fpr_at_95_tpr
        results_dict[method]['AUROC'] = auroc
        results_dict[method]['AUPR'] = aupr

    # Create DataFrame from the results dictionary
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['Method', 'FPR95TPR', 'AUROC', 'AUPR']

    # Save results to a single file
    result_dir = input_dir
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f'final_ood_detection_results.csv')
    df.to_csv(output_file, index=False)

    # Create dictionaries for returning results
    fpr95tpr_dict = {method: results['FPR95TPR'] for method, results in results_dict.items()}
    auroc_dict = {method: results['AUROC'] for method, results in results_dict.items()}
    aupr_dict = {method: results['AUPR'] for method, results in results_dict.items()}

    return fpr95tpr_dict, auroc_dict, aupr_dict
