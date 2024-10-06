import os
import os.path as osp

import argparse
import yaml
import random

import torch
import torch.nn as nn

import math
import time
import matplotlib.pyplot as plt
from addict import Dict #from attrdict import AttrDictfrom tqdm import tqdm
from copy import deepcopy

from data.image import img_to_task, task_to_img, coord_to_img
from data.emnist import EMNIST

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'eval', 'plot', 'ensemble'], default='train', help='Specifies the mode in which the script should run')
    parser.add_argument('--expid', type=str, default='trial', help='Identifier for the experiment')
    parser.add_argument('--resume', action='store_true', default=False, help='Flag to resume training from the last checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='Specifies which GPU to use')

    parser.add_argument('--max_num_points', type=int, default=200, help='Maximum number of points to use in the task')
    parser.add_argument('--class_range', type=int, nargs='*', default=[0, 10], help='Range of classes to use')

    parser.add_argument('--model', type=str, default='cnp', help='Specifies the model to use')
    parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--train_num_samples', type=int, default=4, help='Number of samples for training')
    parser.add_argument('--train_target_all', action='store_true', default=False, help='Flag to indicate target_all for training')

    parser.add_argument('--max_ctx_points', type=int, default=None, help='Maximum number of context points')

    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency of evaluation during training (in epochs)')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving checkpoints (in epochs)')

    parser.add_argument('--eval_seed', type=int, default=42, help='Seed for evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--eval_num_samples', type=int, default=50, help='Number of samples for evaluation')
    parser.add_argument('--eval_logfile', type=str, default=None, help='Log file for evaluation results')
    parser.add_argument('--eval_target_all', action='store_true', default=False, help='Flag to indicate target_all for evaluation')

    parser.add_argument('--plot_seed', type=int, default=None, help='Seed for plotting')
    parser.add_argument('--plot_batch_size', type=int, default=16, help='Batch size for plotting')
    parser.add_argument('--plot_num_samples', type=int, default=30, help='Number of samples for plotting')
    parser.add_argument('--plot_num_ctx', type=int, default=100, help='Number of context points for plotting')
    parser.add_argument('--plot_random_samples', action='store_true', default=False, help='Flag to indicate random samples for plotting')


    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None, help='Noise level for out-of-distribution (OOD) settings')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/emnist/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'emnist', args.model, args.expid)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)
    elif args.mode == 'ensemble':
        ensemble(args, model)

def train(args, model):
    if not osp.isdir(args.root):
        os.makedirs(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = EMNIST(train=True, class_range=args.class_range)
    eval_ds = EMNIST(train=False, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for (x, _) in tqdm(train_loader):

            batch = img_to_task(x,
                    max_num_points=args.max_num_points,max_ctx_points=args.max_ctx_points,target_all=args.train_target_all,
                    device='cuda')


            optimizer.zero_grad()
            outs = model(batch, num_samples=args.train_num_samples)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model,epoch) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = Dict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model,args.num_epochs)

def gen_evalset(args):
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=4)

    batches = []
    for x, _ in tqdm(eval_loader):
        batches.append(img_to_task(x,
            t_noise=args.t_noise,
            max_num_points=args.max_num_points,max_ctx_points=args.max_ctx_points,target_all=args.eval_target_all))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'emnist')
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))

def eval(args, model,epoch=None):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
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

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches):
            for key, val in batch.items():
                batch[key] = val.cuda()
            outs = model(batch, num_samples=args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    c1, c2 = args.class_range
    line = f'{args.model}:{args.expid} {c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    plot(args, model, eval_batches, suffix=epoch if epoch is not None else '')

    return line





def plot_images(args, image_pairs, suffix=''):
    # Determine the number of rows and columns for the subplot grid
    nrows = max(args.plot_batch_size // 4, 1)
    ncols = min(4, args.plot_batch_size)

    fig, axes = plt.subplots(nrows, ncols * 3, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, (image_pair) in enumerate(image_pairs):
        origin, ctx, complete_image = image_pair

        # Convert tensors to numpy arrays
        # origin_np = origin.squeeze().cpu().numpy()
        # ctx_np = ctx.squeeze().cpu().numpy()
        # complete_image_np = complete_image.squeeze().cpu().numpy()

        origin_np = np.clip(origin.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)
        ctx_np = np.clip(ctx.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)
        complete_image_np = np.clip(complete_image.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)

        # Plot the original image
        ax = axes[3 * i]
        ax.imshow(origin_np)
        ax.set_title(f'Origin Image {i}')
        ax.axis('off')

        # Plot the original image
        ax = axes[3 * i + 1]
        ax.imshow(ctx_np)
        ax.set_title(f'Contex Image {i}')
        ax.axis('off')

        # Plot the complete image
        ax = axes[3 * i + 2]
        ax.imshow(complete_image_np)
        ax.set_title(f'Complete Image {i}')
        ax.axis('off')

    plt.tight_layout()

    image_name = osp.join(args.root, f'plot_{args.eval_seed}_{suffix}.png')
    plt.savefig(image_name)
    print(f"{args.model}:Saved image to {image_name}")
    plt.show()
    if args.mode != 'plot':
        plt.close()


def plot(args, model, batch=None, suffix=''):
    if batch is None:
        if args.eval_seed is not None:
            torch.manual_seed(args.eval_seed)
            torch.cuda.manual_seed(args.eval_seed)

        eval_ds = EMNIST(train=False, class_range=args.class_range)
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.plot_batch_size,
                shuffle=False, num_workers=4)


        if args.plot_random_samples:
            batches = list(eval_loader)

            # Generate a random index
            random_index = random.randint(0, len(batches) - 1)

            # Select the random batch
            batch = batches[random_index]

            # batch = next(iter(eval_loader))
            batch = img_to_task(batch[0], max_num_points=args.max_num_points,max_ctx_points=args.max_ctx_points,target_all=args.eval_target_all, device='cuda')
        else:
            batch = next(iter(eval_loader))
            batch = img_to_task(batch[0], max_num_points=args.max_num_points,max_ctx_points=args.max_ctx_points,target_all=args.eval_target_all, device='cuda')

    if args.mode == 'plot':
        suffix = "ckpt"
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        model.eval()
        with torch.no_grad():
            outs = model(batch, num_samples=args.eval_num_samples)
            print(f'ctx_ll {outs.ctx_ll.item():.4f}, tar_ll {outs.tar_ll.item():.4f}')

    # print("batch len", batch[0:5])
    #xp = torch.linspace(-2, 2, 200).cuda()
    # with torch.no_grad():
    #     py = model.predict(batch['xc'], batch['yc'], xp.unsqueeze(0).repeat(args.plot_num_samples, 1).unsqueeze(-1))
    #     mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

    # xp = torch.linspace(-2, 2, 200).cuda()
    images = []
    with torch.no_grad():
        if isinstance(batch, list):
            for b in range(args.plot_batch_size):
                if args.plot_random_samples:
                    bb = random.randint(0, len(batch) - 1)
                    bi = random.randint(0, len(batch[bb]['xc']) - 1)
                else:
                    bb = b % len(batch)
                    bi = b // len(batch)


                py = model.predict(batch[bb]['xc'][bi:bi+1], batch[bb]['yc'][bi:bi+1],batch[bb]['xt'][bi:bi+1], num_samples=args.plot_num_samples)
                yt = py.mean

                task_img, comp_img = task_to_img(batch[bb]['xc'][bi:bi+1], batch[bb]['yc'][bi:bi+1], batch[bb]['xt'][bi:bi+1],yt[0],(1,28,28))

                images.append((task_img, comp_img))
        else:
            origin_img = coord_to_img(batch.x, batch.y, (1, 28, 28))

            py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.plot_num_samples)
            yt = py.mean

            ctx_img, comp_img = task_to_img(batch.xc, batch.yc,
                                             batch.xt, yt[0], (1, 28, 28))

            for i in range(ctx_img.shape[0]):
                images.append((origin_img[i], ctx_img[i], comp_img[i]))
        plot_images(args, images, suffix=suffix)




def ensemble(args, model):
    num_runs = 5
    models = []
    for i in range(num_runs):
        model_ = deepcopy(model)
        ckpt = torch.load(osp.join(results_path, 'emnist', args.model, f'run{i+1}', 'ckpt.tar'))
        model_.load_state_dict(ckpt['model'])
        model_.cuda()
        model_.eval()
        models.append(model_)

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    ravg = RunningAverage()
    with torch.no_grad():
        for batch in tqdm(eval_batches):
            for key, val in batch.items():
                batch[key] = val.cuda()

            ctx_ll = []
            tar_ll = []
            for model in models:
                outs = model(batch,
                        num_samples=args.eval_num_samples,
                        reduce_ll=False)
                ctx_ll.append(outs.ctx_ll)
                tar_ll.append(outs.tar_ll)

            if ctx_ll[0].dim() == 2:
                ctx_ll = torch.stack(ctx_ll)
                tar_ll = torch.stack(tar_ll)
            else:
                ctx_ll = torch.cat(ctx_ll)
                tar_ll = torch.cat(tar_ll)

            ctx_ll = logmeanexp(ctx_ll).mean()
            tar_ll = logmeanexp(tar_ll).mean()

            ravg.update('ctx_ll', ctx_ll)
            ravg.update('tar_ll', tar_ll)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    filename = f'ensemble_{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, filename), mode='w')
    logger.info(ravg.info())
if __name__ == '__main__':
    main()

