import os
import os.path as osp
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from addict import Dict
from copy import deepcopy
from torch.distributions import StudentT
from data.image import coord_to_img, task_to_img
from data.emnist import EMNIST
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage
from utils.misc import load_module, logmeanexp
import datetime
import re
from tqdm import tqdm
from exp_image import lower_half_to_task, upper_half_to_task
from exp_logger import get_logger


exp_logger = None
def get_max_expid(root_path):
    max_expid = -1
    exp_dir = osp.join(root_path, 'experiments', 'emnist')
    if osp.exists(exp_dir):
        for folder_name in os.listdir(exp_dir):
            match = re.match(r'experiment_(\d+)', folder_name)
            if match:
                expid = int(match.group(1))
                if expid > max_expid:
                    max_expid = expid
    return max_expid


def load_model_checkpoint(model_name, exp_id, results_path):
    model_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', f'{model_name}.py'))
    config_file_path = os.path.abspath \
        (os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'emnist', f'{model_name}.yaml'))

    model_cls = getattr(load_module(model_file_path), model_name.upper())
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    ckpt_path = osp.join(results_path, 'emnist', model_name, exp_id, 'ckpt.tar')
    ckpt = torch.load(ckpt_path)
    exp_logger.info(f"Loaded model {model_name} from {ckpt_path}")
    # TODO: add logger and create log file

    model.load_state_dict(ckpt.model)

    return model


def plot_images2(args, images, suffix=''):
    """
    Plot comparison of predictions from multiple models for each origin image and context image.

    Parameters:
    - args: Argument object with plotting configurations.
    - images: List of tuples ( (origin_image, context_image), model_samples )
              where model_samples is a list of lists containing samples from different models.
    - suffix: Optional suffix for saving the image file.
    """
    # Determine the number of rows (one for each model) and columns (origin + context + samples for each model)
    nrows = len(images[0][1])  # Number of models
    ncols = 2 + len(images[0][1][0])  # 2 (origin and context) + number of samples per model

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, ((origin, ctx), model_samples) in enumerate(images):
        # Convert tensors to numpy arrays and clip for valid image display
        origin_np = origin.squeeze().cpu().numpy()
        ctx_np = ctx.squeeze().cpu().numpy()

        # Handle grayscale or RGB images
        if origin_np.ndim == 3:  # RGB image
            origin_np = np.clip(origin_np.transpose(1, 2, 0), 0, 1)
        else:  # Grayscale image
            origin_np = np.clip(origin_np, 0, 1)

        if ctx_np.ndim == 3:  # RGB image
            ctx_np = np.clip(ctx_np.transpose(1, 2, 0), 0, 1)
        else:  # Grayscale image
            ctx_np = np.clip(ctx_np, 0, 1)

        # Plot the original image
        for j, samples in enumerate(model_samples):
            ax = axes[ncols * j]
            ax.imshow(origin_np, cmap='gray' if origin_np.ndim == 2 else None)
            ax.set_title(f'Origin Image', fontsize=20, pad=20)
            ax.axis('off')

            # Plot the context image
            ax = axes[ncols * j + 1]
            ax.imshow(ctx_np, cmap='gray' if ctx_np.ndim == 2 else None)
            ax.set_title(f'Context Image', fontsize=20, pad=20)
            ax.axis('off')

            # Plot model predictions for all samples
            for k, sample_batch in enumerate(samples):
                # Loop over the batch of images (if the batch size is > 1)
                #for idx, sample in enumerate(sample_batch):
                sample_np = sample_batch[0].squeeze().cpu().numpy()

                # Handle grayscale or RGB samples
                if sample_np.ndim == 3:  # RGB image
                    sample_np = np.clip(sample_np.transpose(1, 2, 0), 0, 1)
                else:  # Grayscale image
                    sample_np = np.clip(sample_np, 0, 1)

                ax = axes[ncols * j + 2 + k]
                ax.imshow(sample_np, cmap='gray' if sample_np.ndim == 2 else None)
                ax.set_title(f'Model {args.models[j]} Sample {k}', fontsize=20, pad=20)
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)

        image_name = osp.join(args.exp_path, f'plot_{args.pattern_name}_{i}.png')

        # Save the image to file
        plt.savefig(image_name)
        exp_logger.info(f"Models {', '.join(args.models)}: Saved comparison image to {image_name}")
        plt.show()


def plot_images(args, images, suffix=''):
    """
    Plot comparison of predictions from multiple models for each origin image and context image.
    Additionally, save all images combined in a single file as 'plot_{args.pattern_name}_all.png'.

    Parameters:
    - args: Argument object with plotting configurations.
    - images: List of tuples ( (origin_image, context_image), model_samples )
              where model_samples is a list of lists containing samples from different models.
    - suffix: Optional suffix for saving the image file.
    """
    # Determine the number of rows (one for each model) and columns (origin + context + samples for each model)
    nrows = len(images[0][1])  # Number of models
    ncols = 2 + len(images[0][1][0])  # 2 (origin and context) + number of samples per model

    # List to store all figures for combined plot later
    all_figures = []

    for i, ((origin, ctx), model_samples) in enumerate(images):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

        # Convert tensors to numpy arrays and clip for valid image display
        origin_np = origin.squeeze().cpu().numpy()
        ctx_np = ctx.squeeze().cpu().numpy()

        # Handle grayscale or RGB images
        if origin_np.ndim == 3:  # RGB image
            origin_np = np.clip(origin_np.transpose(1, 2, 0), 0, 1)
        else:  # Grayscale image
            origin_np = np.clip(origin_np, 0, 1)

        if ctx_np.ndim == 3:  # RGB image
            ctx_np = np.clip(ctx_np.transpose(1, 2, 0), 0, 1)
        else:  # Grayscale image
            ctx_np = np.clip(ctx_np, 0, 1)

        # Plot the original and context images, then model predictions
        for j, samples in enumerate(model_samples):
            ax = axes[ncols * j]
            ax.imshow(origin_np, cmap='gray' if origin_np.ndim == 2 else None)
            ax.set_title(f'Origin Image {i}', fontsize=20, pad=20)
            ax.axis('off')

            ax = axes[ncols * j + 1]
            ax.imshow(ctx_np, cmap='gray' if ctx_np.ndim == 2 else None)
            ax.set_title(f'Context Image {i}', fontsize=20, pad=20)
            ax.axis('off')

            for k, sample_batch in enumerate(samples):
                sample_np = sample_batch[0].squeeze().cpu().numpy()

                if sample_np.ndim == 3:  # RGB image
                    sample_np = np.clip(sample_np.transpose(1, 2, 0), 0, 1)
                else:  # Grayscale image
                    sample_np = np.clip(sample_np, 0, 1)

                ax = axes[ncols * j + 2 + k]
                ax.imshow(sample_np, cmap='gray' if sample_np.ndim == 2 else None)
                ax.set_title(f'Model {args.models[j]} Sample {k}', fontsize=20, pad=20)
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)

        image_name = osp.join(args.exp_path, f'plot_{args.pattern_name}_{i}.png')

        # Save the individual image
        plt.savefig(image_name)
        exp_logger.info(f"Models {', '.join(args.models)}: Saved comparison image to {image_name}")
        plt.show()

        # Add the current figure to the list of all figures for combined saving later
        all_figures.append(fig)

    # Save the combined plot of all images in one file
    combined_fig, combined_axes = plt.subplots(nrows=len(all_figures) * nrows, ncols=ncols,
                                               figsize=(5 * ncols, 5 * len(all_figures) * nrows))
    combined_axes = combined_axes.flatten()

    # Add all images into the combined plot
    for i, fig in enumerate(all_figures):
        for j, ax_src in enumerate(fig.get_axes()):
            ax_dst = combined_axes[ncols * nrows * i + j]
            ax_dst.imshow(ax_src.images[0].get_array())
            ax_dst.set_title(ax_src.get_title(), fontsize=20, pad=20)
            ax_dst.axis('off')

    combined_image_name = osp.join(args.exp_path, f'plot_{args.pattern_name}_all.png')

    # Save the combined image
    plt.tight_layout()
    plt.savefig(combined_image_name)
    exp_logger.info(f"Saved combined comparison image to {combined_image_name}")
    plt.show()


def compare_and_plot_patterns(args):
    pattern_funcs = {
        'upper_half': upper_half_to_task,
        'lower_half': lower_half_to_task,
        # Add other patterns here if needed
    }

    if args.pattern_name not in pattern_funcs:
        raise ValueError(f"Unknown pattern name: {args.pattern_name}")

    pattern_func = pattern_funcs[args.pattern_name]

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.plot_batch_size, shuffle=True, num_workers=2)

    models = {}
    for model_name in args.models:
        model = load_model_checkpoint(model_name, args.ckptid, results_path)
        models[model_name] = model

    images = []
    for i in tqdm(range(args.num_images_to_compare)):
        # print(f"Generating task. image {i}")
        batch = next(iter(eval_loader))

        batch = pattern_func(batch[0], device='cuda')

        origin_img = coord_to_img(batch.x, batch.y, (1, 28, 28))
        # print("batch.x.shape", batch.x.shape)
        cxt_img = None
        models_samples = []

        for model_name,model in models.items():
            samples = []
            # print(f"Loading model {model_name}")
            # model = load_model_checkpoint(model_name, args.chptid, results_path)
            with torch.no_grad():
                py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.plot_num_samples)
                yt = py.mean
                # yt.shape torch.Size([5, 16, 392, 1])
                for sample_num in range(args.plot_num_samples):

                    _, comp_img = task_to_img(batch.xc, batch.yc, batch.xt, yt[sample_num], (1, 28, 28))
                    # comp_img.shape torch.Size([16, 3, 28, 28])
                    if cxt_img is None:
                        cxt_img = _

                    samples.append(comp_img)
            models_samples.append(samples)

        images.append(((origin_img[0], cxt_img[0]), models_samples))

    plot_images(args, images)


def main():
    global exp_logger
    parser = argparse.ArgumentParser(description="Compare and plot models on EMNIST dataset using specific patterns")
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of model names')
    parser.add_argument('--eval_seed', type=int, default=42, help='Seed for evaluation')
    parser.add_argument('--plot_num_samples', type=int, default=5, help='Number of samples to plot')
    parser.add_argument('--plot_batch_size', type=int, default=16, help='')
    parser.add_argument('--class_range', type=int, nargs=2, default=[0, 10], help='Range of classes to use')
    parser.add_argument('--max_num_points', type=int, default=200, help='Maximum number of points to use in the task')
    parser.add_argument('--max_ctx_points', type=int, default=100, help='Maximum number of context points')
    parser.add_argument('--target_all', action='store_true', help='Flag to indicate target_all for evaluation')
    parser.add_argument('--num_images_to_compare', type=int, default=5, help='Number of images to compare')
    parser.add_argument('--pattern_name', type=str, default="upper_half", help='Name of the pattern function to use')
    parser.add_argument('--expid', type=str, default='', help='Experiment ID')
    parser.add_argument('--ckptid', type=str, default='run1', help='Checkpoint ID')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.expid == '':
        max_expid = get_max_expid(results_path)
        args.expid = str(max_expid + 1) if max_expid >= 0 else '0'

    args.exp_path = osp.join(results_path, 'experiments', 'emnist', f"experiment_{args.expid}")

    if not osp.isdir(args.exp_path):
        os.makedirs(args.exp_path)

    exp_logger = get_logger(log_dir=args.exp_path)

    with open(osp.join(args.exp_path, 'exp_args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    compare_and_plot_patterns(args)


if __name__ == '__main__':
    main()
