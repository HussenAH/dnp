import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', ))
sys.path.append(project_root)

import os.path as osp
import argparse
import yaml
import random
import torch
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt
from addict import Dict
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage
from regression.data.era5 import ERA5Dataset

import regression.utils.helper as era5_utils

import regression.data.era5 as dataset
from regression.utils.helper import plot_inference_2d  # Import the plot function
from regression.data.era5 import era5_to_task, task_to_era5
from regression.utils.helper import plot_inference_2d,plot_temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'plot', 'ensemble'], default='train',
                        help='Specifies the mode in which the script should run')
    parser.add_argument('--next_mode', choices=['train', 'eval', 'plot', 'ensemble', 'stop'], default='eval',
                        help='Specifies the next mode in which the script should run')
    parser.add_argument('--expid', type=str, default='trial', help='Identifier for the experiment')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Flag to resume training from the last checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='Specifies which GPU to use')

    parser.add_argument('--model', type=str, default='cnp', help='Specifies the model to use')
    parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--train_num_samples', type=int, default=4, help='Number of samples for training')
    parser.add_argument('--train_target_all', action='store_true', default=False,
                        help='Flag to indicate target_all for training')

    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--eval_freq', type=int, default=100, help='Frequency of evaluation during training (in epochs)')
    parser.add_argument('--save_freq', type=int, default=100, help='Frequency of saving checkpoints (in epochs)')

    parser.add_argument('--group', type=str, required=False, help='Group')

    parser.add_argument('--n_x_axis', type=int, required=False, help='Number of grid points per axis')
    parser.add_argument('--batch_size', type=int, default=100, required=False, help='Batch size.')
    parser.add_argument('--n_iterat_per_epoch', type=int, default=250, required=False,
                        help='Number of iterations per epoch.')
    parser.add_argument('--filename', type=str, required=False, help='Filename for saving model.')
    parser.add_argument('--length_scale_in', type=float, required=False, help='Length scale for encoder.')
    parser.add_argument('--seed', type=int, required=False, help='Seed for randomness.')
    parser.add_argument('--shape_reg', type=float, required=False, help='Shape Regularizer')
    parser.add_argument('--n_val_samples', type=int, required=False, help='Number of validation samples.')
    parser.add_argument('--n_eval_samples', type=int, required=False,
                        help='Number of evaluation samples after training.')
    parser.add_argument('--testing_group', type=str, required=False,
                        help='Group with respect to which equivariance is tested.')
    parser.add_argument('--n_data_passes', type=int, required=False, help='Passes through data used for evaluation.')
    parser.add_argument('--plot_batch_size', type=int, default=16, help='Batch size for plotting')
    parser.add_argument('--eval_seed', type=int, default=42, help='Seed for evaluation')
    parser.add_argument('--plot_random_samples', action='store_true', default=False, help='Flag to indicate random samples for plotting')
    parser.add_argument('--eval_num_samples', type=int, default=50, help='Number of samples for evaluation')
    parser.add_argument('--plot_num_samples', type=int, default=30, help='Number of samples for plotting')
    parser.add_argument('--max_ctx_points', type=int, default=None, help='Maximum number of context points')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/era5/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'era5', args.model, args.expid)

    print(f"Number of parameters: {era5_utils.count_parameters(model, print_table=False)}")

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)


def train(args, model):
    if not osp.isdir(args.root):
        os.makedirs(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    #TODO: add as args
    min_ctx_points = 2
    args.max_ctx_points = (41 * 41) - 1 if args.max_ctx_points is None else args.max_ctx_points

    train_ds = ERA5Dataset('train', min_ctx_points, args.max_ctx_points, place='US', normalize=True, circular=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("len(train_load)", len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    if args.resume:
        # TODO: check if this works
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logfilename = ckpt['logfilename']
        start_epoch = ckpt['epoch']
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_loss = 0.0

        model.train()
        # for i in range(args.n_iterat_per_epoch):
        for b in tqdm(train_loader):
            # print(x.shape)
            # x_context, y_context, x_target, y_target = train_ds.get_rand_batch(batch_size=args.batch_size, cont_in_target=True)
            # x_context, y_context, x_target, y_target = x_context.to('cuda'), y_context.to('cuda'), x_target.to('cuda'), y_target.to('cuda')
            # print("x_context", x_context.shape)
            # print("y_context", y_context.shape)
            # print("x_target", x_target.shape)
            # print("y_target", y_target.shape)

            batch = era5_to_task(b, device=device)
            # print(batch)
            # print("batch.x.shape", batch.x.shape)
            # print("batch.y.shape", batch.y.shape)
            # print("batch.xc.shape", batch.xc.shape)
            # print("batch.yc.shape", batch.yc.shape)
            # print("batch.xt.shape", batch.xt.shape)
            # print("batch.yt.shape", batch.yt.shape)

            optimizer.zero_grad()
            outs = model(batch, num_samples=args.train_num_samples)
            loss = outs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            for key, val in outs.items():
                ravg.update(key, val)
        # logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss / args.n_iterat_per_epoch:.4f}")

        line = f'{args.model}:{args.expid} epoch {epoch}/{args.num_epochs} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)


        if args.next_mode != 'stop' and epoch % args.eval_freq == 0:
            logger.info(eval(args, model,epoch) + '\n')

        ravg.reset()

        # if (epoch + 1) % args.save_freq == 0:
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'logfilename': logfilename,
        #         'loss': running_loss,
        #     }, osp.join(args.root, 'ckpt.tar'))

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = Dict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))
            logger.info(f"Checkpoint saved at epoch {epoch}")

    if args.next_mode != 'stop':
        args.mode = 'eval'
        eval(args, model)


def gen_dataset(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    # Load the dataset
    eval_ds = ERA5Dataset(mode='valid')
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # Process the dataset into batches
    batches = []
    for x_context, y_context, x_target, y_target in tqdm(eval_loader):
        batch = {
            'x_context': x_context,
            'y_context': y_context,
            'x_target': x_target,
            'y_target': y_target
        }
        batches.append(batch)

    # Reset the random seed
    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    # Save the processed dataset to a file
    path = osp.join(evalsets_path, 'era5')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = 'era5_eval_dataset.tar'
    torch.save(batches, osp.join(path, filename))


def eval(args, model, epoch=None):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt['model_state_dict'])
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f'eval_{c1}-{c2}'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    val_ds = ERA5Dataset('valid', 2, 50, place='US', normalize=True, circular=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()
    total_log_ll = 0.0
    with torch.no_grad():
        for _ in range(args.n_eval_samples):
            x_context, y_context, x_target, y_target = val_ds.get_rand_batch(batch_size=args.batch_size,
                                                                             cont_in_target=False)
            x_context, y_context, x_target, y_target = x_context.to('cuda'), y_context.to('cuda'), x_target.to(
                'cuda'), y_target.to('cuda')

            batch = Dict(xc=x_context, yc=y_context, x=x_target, y=y_target)
            outs = model(batch)
            log_ll = outs.tar_ll
            total_log_ll += log_ll.item()

    if logger:
        logger.info(f"Validation Log Likelihood: {total_log_ll / args.n_eval_samples:.4f}")


def plot(args, model, batch=None, suffix=''):
    torch.multiprocessing.set_sharing_strategy('file_system')
    eval_ds = None
    if batch is None:
        if args.eval_seed is not None:
            torch.manual_seed(args.eval_seed)
            torch.cuda.manual_seed(args.eval_seed)

        args.max_ctx_points = (41 * 41) - 1 if args.max_ctx_points is None else args.max_ctx_points

        eval_ds = ERA5Dataset('valid', 2, args.max_ctx_points, place='US', normalize=True, circular=True)
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.plot_batch_size, shuffle=False, num_workers=4)

        if args.plot_random_samples:
            batches = list(eval_loader)
            random_index = random.randint(0, len(batches) - 1)
            batch = batches[random_index]
        else:
            batch = next(iter(eval_loader))

        batch = era5_to_task(batch, max_ctx_points=args.max_ctx_points, device='cuda')

    print("batch.x.shape", batch.x.shape)
    print("batch.y.shape", batch.y.shape)
    print("batch.xc.shape", batch.xc.shape)
    print("batch.yc.shape", batch.yc.shape)
    print("batch.xt.shape", batch.xt.shape)
    print("batch.yt.shape", batch.yt.shape)

    if args.mode == 'plot':
        print("Loading model from checkpoint")
        suffix = "ckpt"
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        model.eval()
        # with torch.no_grad():
        #     outs = model(batch, num_samples=args.eval_num_samples)
        #     print(f'ctx_ll {outs.ctx_ll.item():.4f}, tar_ll {outs.tar_ll.item():.4f}')

    print("model has been loaded")

    results = []
    with torch.no_grad():
        if isinstance(batch, list):
            print("batch is a list")
            for b in range(args.plot_batch_size):
                if args.plot_random_samples:
                    bb = random.randint(0, len(batch) - 1)
                    bi = random.randint(0, len(batch[bb]['xc']) - 1)
                else:
                    bb = b % len(batch)
                    bi = b // len(batch)

                py = model.predict(batch[bb]['xc'][bi:bi+1], batch[bb]['yc'][bi:bi+1], batch[bb]['xt'][bi:bi+1], num_samples=args.plot_num_samples)
                yt = py.mean
                context, concat = task_to_era5(batch[bb])
                results.append((context, concat))
        else:
            print("batch is not a list")

            py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.plot_num_samples)
            print("py",py)
            print("py.mean.shape",py.mean.shape)
            print("py.variance.shape",py.variance.shape)
            # yt = py.mean
            # context, concat = task_to_era5(batch)

            Means, Covs = py.mean, py.variance

            # print("Means",Means)
            # print("Covs",Covs)
            print("Means[0]",Means[0].shape)
            print("Covs[0]",Covs[0].shape)
            mean_0 = Means[0]




            # Means, Covs = self.forward(X_Context, Y_Context, X_Target)
            # Plot predictions against ground truth:
            # for i in range(X_Context.size(0)):
            #     plot_inference_2d(X_Context[i], Y_Context[i], X_Target[i], Y_Target[i], Predict=Means[i].detach(),
            #                       Cov_Mat=Covs[i].detach(), title=title)
            #
            # for i in range(context.x.shape[0]):
            #     results.append((origin[i], context, concat))

            # plot_inference_2d(args, results, suffix=suffix)
            # plot_inference_2d(batch.xc[0], batch.yc[0,:,[2,3]], X_Target=batch.xt[0], Y_Target=batch.yt[0,:,[2,3]], Predict=Means[0], Cov_Mat=Covs[0], title="",
            #                   size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False)

            # In your plotting code
            plot_inference_2d(batch.xc[0].cpu(), batch.yc[0, :, [2, 3]].cpu(),
                              X_Target=batch.xt[0].cpu(), Y_Target=batch.yt[0, :, [2, 3]].cpu(),
                              Predict=mean_0[0,:,[2,3]].detach().cpu(), Cov_Mat=Covs[0][0,:,[2,3]].detach().cpu(),
                              title="", size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False,out_path=args.root)

            """
            # Translate the data back to original scale
            """
            batch.x,batch.y = eval_ds.translater.translate_to_original_scale(batch.x.cpu(), batch.y.cpu())
            batch.xc,batch.yc = eval_ds.translater.translate_to_original_scale(batch.xc.cpu(), batch.yc.cpu())
            batch.xt,batch.yt = eval_ds.translater.translate_to_original_scale(batch.xt.cpu(), batch.yt.cpu())
            _, mean_0 = eval_ds.translater.translate_to_original_scale(batch.xt.cpu(), Means[0].cpu())

            plot_temperature(batch.xc[0].cpu(), batch.yc[0].cpu(),batch.xt[0].cpu(), batch.yt[0].cpu(),mean_0[0,:,:].detach().cpu(),save_path=args.root+"/temperature_plot.png")
            # plot_inference_2d(X_Context, Y_Context, X_Target=None, Y_Target=None, Predict=None, Cov_Mat=None, title="",
            #                   size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False)

def ensemble(args, model):
    logger = get_logger('era5')  # Initialize the logger

    if args.data_set == 'small':
        path_to_val_file = 'path_to_small_val_file'
    elif args.data_set == 'big':
        path_to_val_file = 'path_to_big_val_file'
    else:
        raise ValueError("Unknown data set type")

    val_ds = ERA5Dataset(path_to_val_file, 2, 50, place='US', normalize=True, circular=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()
    total_log_ll = 0.0
    with torch.no_grad():
        for _ in range(args.n_eval_samples):
            x_context, y_context, x_target, y_target = val_ds.get_rand_batch(batch_size=args.batch_size,
                                                                             cont_in_target=False)
            x_context, y_context, x_target, y_target = x_context.to('cuda'), y_context.to('cuda'), x_target.to(
                'cuda'), y_target.to('cuda')

            batch = Dict(xc=x_context, yc=y_context, x=x_target, y=y_target)
            outs = model(batch)
            log_ll = outs.tar_ll
            total_log_ll += log_ll.item()

    if logger:
        logger.info(f"Ensemble Log Likelihood: {total_log_ll / args.n_eval_samples:.4f}")


if __name__ == '__main__':
    main()

