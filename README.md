# MADOD

This repository contains the implementation of MADOD (Meta-learned Across Domain Out-of-distribution Detection), as described in the paper *"MADOD: Generalizing OOD Detection to Unseen Domains via G-Invariance Meta-Learning."*

## Requirements

Ensure you have the following installed:

```
Python: 3.9.13
PyTorch: 1.12.1
Torchvision 0.13.1
CUDA 11.6
CUDNN 8302
NumPy 1.23.1
PIL 9.2.0
```

For the complete list of dependencies, refer to the `requirements.txt` file.

## Quick Start

### Dataset Download

Before training the model, you need to download the necessary datasets. A convenient script is provided for this. Run the following command to download all required datasets:

```
python -m domainbed.scripts.download
```


### Model Training

To train the MADOD model, execute the following command:

```
python -m domainbed.scripts.train_meta\
         --dataset PACS\
         --ood_classes 6\
         --gpu 0\
         --test_envs 0\
         --n_tasks 4\
         --n_pseudo 1\
         --k_support 5\
         --k_query 5\
         --k_pseudo 5\
         --k_adapt 5
```


### Resuming Training from Checkpoint

If you wish to resume training from a checkpoint, run this command:
```
python -m domainbed.scripts.train_meta\
         --checkpoint_dir=domainbed/sweep_outputs/PACS/MADOD0\
         --dataset PACS\
         --ood_classes 6\
         --gpu 0\
         --test_envs 0\
         --n_tasks 4\
         --n_pseudo 1\
         --k_support 5\
         --k_query 5\
         --k_pseudo 5\
         --k_adapt 5
```


### Performance Summary

To summarize the OOD detection performance, use the following command:
```
python -m domainbed.scripts.meta_results_summary0\
         --input_dir=domainbed/sweep_outputs/PACS/MADOD0\
         --dataset PACS\
         --ood_classes 3 6
```
>>>>>>> 45c722d (initial commit)
