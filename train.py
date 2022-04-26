import argparse
import os
import random

import dllogger as DLLogger
import numpy as np
import torch
import torch.distributed as dist
from dllogger import StdOutBackend, Verbosity
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.sep_dataset import SeparationDataset
from losses.loss_switcher import get_loss_fn
from models.model_switcher import get_model
from utils.common_utils import get_statistics, get_datetime_ref, save_checkpoint, \
    get_last_checkpoint_filename, load_checkpoint, ParseFromConfigFile
from utils.tb_logger import TBLogger as Logger

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def parse_args(parser):
    """
    Parse from config file.
    Add other arguments if needed
    """
    parser.add_argument('--config-file', action=ParseFromConfigFile,
                        type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu-run', action='store_true')
    return parser


def init_distributed(rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank)

    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

    print("Done initializing distributed")


def evaluate(args, model, eval_loader, distributed_run, tb_logger, iteration):
    model.eval()
    if distributed_run:
        model = model.module
    val_loss = []
    display_idx = 0 if args.debug else random.choice(range(len(eval_loader)))
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            mixture_wav, target_wav = [t.to(args.device, non_blocking=True) for t in batch]

            output, target = model(mixture_wav, target_wav, patch_length=args.val_patch_length)
            loss = torch.nn.L1Loss()(output, target)
            val_loss.append(loss.item())
            if display_idx == batch_idx:
                x_wav = mixture_wav[0, :1].detach()
                y_wav = target_wav[0, :, :1].detach()
                if len(output.shape) == 4:
                    y_hat_wav = output[0, :, :1].detach()
                elif len(output.shape) == 5:
                    y_hat_mag = output[0, 0, :1].detach().transpose(1, 2)
                    y_stft = model.time_domain_wrapper.transform(y_wav[0], patch_length=args.val_patch_length,
                                                                  return_magnitude=False)
                    y_hat_phase = y_stft.angle()
                    y_hat_wav = model.time_domain_wrapper.inverse_transform(magnitude=y_hat_mag, phase=y_hat_phase)
            if args.debug:
                break
        tb_logger.log_audio(x_wav.cpu(), y_wav.cpu(), y_hat_wav.cpu(), args.data_loader_args['train']['target_sources'])

    val_loss = sum(val_loss) / len(val_loss)
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    if args.debug:
        args.exp_name = 'debug'
    args.output = os.path.join(args.exp_dir, args.model_name, args.exp_name)
    if args.use_date_time_ref:
        date_time_ref = get_datetime_ref(args)
        args.output = os.path.join(args.output, date_time_ref)
    args.device = 'cuda' if torch.cuda.is_available() and not args.cpu_run else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = local_rank
        args.world_size = world_size
        args.distributed = True

    else:
        local_rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.distributed = False

    if args.distributed:
        init_distributed(local_rank)

    if local_rank == 0:
        os.makedirs(args.output, exist_ok=True)
        DLLogger.init(backends=[
            StdOutBackend(Verbosity.VERBOSE)])
        tb_log_dir = os.path.join(args.output, 'TB_Logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        TBLogger = Logger(tb_log_dir)
    else:
        DLLogger.init(backends=[])
        TBLogger = None

    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.use_stats:
        if args.stats_path is None or not os.path.exists(args.stats_path):

            import musdb
            mus = musdb.DB(root=args.data_loader_args['train']['data_root'], subsets='train', is_wav=True)
            input_mean, input_scale = get_statistics(args, mus)
            stats_path = os.path.join(args.data_loader_args['train']['data_root'], 'musdb_train.stats')
            if local_rank == 0:
                torch.save(
                    {
                        'mean': input_mean,
                        'std': input_scale
                    },
                    stats_path
                )
            args.stats_path = stats_path
        else:
            stats = torch.load(args.stats_path)
            input_mean = stats['mean']
            input_scale = stats['std']
    else:
        input_mean = None
        input_scale = None

    model = get_model(model_name=args.model_name, model_args=args.model_args,
                      input_mean=input_mean, input_scale=input_scale).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = [0]
    iteration = 0
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler(enabled=args.amp)

    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(args.output, args.model_name)

    if args.checkpoint_path != "":
        model, optimizer, scaler, epoch, iteration, config = load_checkpoint(
            args.checkpoint_path, model, optimizer, scaler, start_epoch)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank],
                    output_device=args.local_rank)

    start_epoch = start_epoch[0]

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay,
    #                                                    last_epoch=start_epoch - 1)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_epoch,
                                                gamma=args.scheduler_gamma, last_epoch=start_epoch - 1)

    criterion = get_loss_fn(args.loss_fn)

    trainset = SeparationDataset(**args.data_loader_args['train'])
    valset = SeparationDataset(**args.data_loader_args['validation'])

    if args.distributed:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=args.num_workers, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True)
    val_loader = None
    if local_rank == 0:
        val_loader = DataLoader(valset, num_workers=8, shuffle=False,
                                sampler=None,
                                batch_size=1, pin_memory=False,
                                drop_last=True)

    model.train()
    best_val_loss = None

    for epoch in range(start_epoch, args.epochs):
        model.train()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            model.train()
            mixture, target = [t.to(args.device, non_blocking=True) for t in batch]

            with torch.cuda.amp.autocast(enabled=args.amp):
                output, target = model(mixture, target)
                with torch.cuda.amp.autocast(enabled=False):
                    loss = criterion(output, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            loss_dict = {
                'it': iteration,
                'train_loss': loss.item(),
            }

            if iteration % args.eval_interval == 0 and local_rank == 0:
                val_loss = evaluate(args, model, val_loader, args.distributed, TBLogger, iteration)
                loss_dict['val_loss'] = val_loss
                if best_val_loss is None or val_loss < best_val_loss:
                    save_checkpoint(iteration, model, optimizer,
                                    scaler, epoch, args, args.output, args.model_name, local_rank,
                                    args.distributed, is_best=True)
                    best_val_loss = val_loss

            if iteration % args.save_interval == 0 and local_rank == 0:
                save_checkpoint(iteration, model, optimizer,
                                scaler, epoch, args, args.output, args.model_name, local_rank, args.distributed)

            DLLogger.log(step=(epoch, i, len(train_loader)), data=loss_dict)
            if local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                TBLogger.log_training(
                    loss_dict, lr, iteration)
            iteration += 1
            if args.debug:
                break

        scheduler.step()

        DLLogger.log(step=(epoch,), data={'End ': epoch})

        if local_rank == 0:
            DLLogger.flush()

        if args.debug:
            break


if __name__ == '__main__':
    main()

    # python train.py --config-file configs/demucs/demucs_config.yaml
    # python -m torch.distributed.run --nproc_per_node=4 train.py --config-file configs/demucs/demucs_config.yaml
