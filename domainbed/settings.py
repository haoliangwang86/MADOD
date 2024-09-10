from domainbed import model_selection

SELECTION_METHODS = [
    # model_selection.IIDAccuracySelectionMethod,
    model_selection.LeaveOneOutSelectionMethod,
    # model_selection.OracleSelectionMethod,
]

OOD_DETECTION_METHODS = ['ocsvm', 'ddu', 'msp', 'energy']

EVAL_METRICS = ['FPR95TPR', 'AUROC', 'AUPR']


def get_dataset_info(dataset):
    if dataset == 'ColoredMNIST':
        num_classes = 3
        n_envs = 3
    elif dataset == 'PACS':
        num_classes = 7
        n_envs = 4
    elif dataset == 'VLCS':
        num_classes = 5
        n_envs = 4
    elif dataset == 'TerraIncognita':
        num_classes = 9
        n_envs = 4
    else:
        print("Dataset not supported!")
        raise NotImplementedError

    return num_classes, n_envs
