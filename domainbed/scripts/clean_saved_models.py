# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import numpy as np

from domainbed.lib.misc import get_the_best_models_configurations
from domainbed.settings import get_dataset_info


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    algorithm = args.algorithm
    dataset = args.dataset

    num_classes, n_envs = get_dataset_info(dataset)

    saved_models = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pkl"):
                saved_models.append(os.path.join(root, file))

    best_models = []
    for test_env in range(n_envs):
        print(f"Test env: {test_env}")
        best_models_configs = get_the_best_models_configurations(input_dir, algorithm, dataset, test_env)

        # mark best models
        for selection_method, best_configs in best_models_configs.items():
            for trial_seed, config in best_configs.items():
                best_models.append(os.path.join(config['args']['output_dir'], f"model_step{config['step']}.pkl"))

    print("Best models that won't be cleaned:")
    print(best_models)

    delete_models = {model for model in saved_models if model not in best_models}

    for model in delete_models:
        if os.path.exists(model):
            os.remove(model)
        else:
            print(f"Model {model} does not exist")

    print("Clean done!")
