"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import tensorboardX
import argparse

from core.utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
from core.trainer import MUNIT_Trainer
from datasets import get_mnist_loaders, get_pacs_vlcs_loaders

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.backends.cudnn as cudnn
import torch
import os
import sys
import shutil

parser = argparse.ArgumentParser(description='PyTorch training')
parser.add_argument('--output_path', type=str, default='saved_models/VLCSOOD',
                        help="Path where images/checkpoints will be saved")
parser.add_argument("--resume", action="store_true",
                        help='Resumes training from last avaiable checkpoint')
parser.add_argument('--dataset', type=str, default="VLCS")
parser.add_argument('--device', default='1', type=str,
                    help='CUDA device index')
parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
parser.add_argument('--ood_classes', type=int, nargs='+', default=[6])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

cudnn.benchmark = True

# Load experiment setting
if args.dataset in ['PACS', 'VLCS']:
    config_path = 'core/big_munit.yaml'
elif args.dataset in ['ColoredMNIST']:
    config_path = 'core/tiny_munit.yaml'
else:
    raise NotImplementedError("Not supported!")

config = get_config(config_path)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = args.output_path

# Setup model and data loader
device = torch.device('cuda')
trainer = torch.nn.DataParallel(MUNIT_Trainer(config))
trainer.to(device)



if args.dataset in ['PACS', 'VLCS']:
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_pacs_vlcs_loaders(args.dataset, args.test_envs, args.ood_classes)
elif args.dataset in ['ColoredMNIST']:
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_mnist_loaders()

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(config_path))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path + "/logs", model_name))
output_directory = os.path.join(args.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(config_path, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.module.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.module.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.module.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
