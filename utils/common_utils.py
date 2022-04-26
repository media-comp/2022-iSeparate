import argparse
import datetime
import glob
import os

from sklearn.preprocessing import StandardScaler
import torchaudio
import yaml
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ParseFromConfigFile(argparse.Action):

    def __init__(self, option_strings, type, dest, help=None, required=False):
        super(ParseFromConfigFile, self).__init__(option_strings=option_strings, type=type, dest=dest, help=help,
                                                  required=required)

    def __call__(self, parser, namespace, values, option_string):

        with open(values, 'r') as f:
            data = yaml.safe_load(f)

        for k, v in data.items():
            setattr(namespace, k, v)


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def preprocess_audio(audio, orig_rate, target_models):
    mono = target_models[list(target_models.keys())[0]]['config'].model_args['audio_channels'] == 1
    target_rate = target_models[list(target_models.keys())[0]]['config'].data_loader_args['train']['sample_rate']
    resample_fn = torchaudio.transforms.Resample(orig_freq=orig_rate, new_freq=target_rate)

    audio = audio.transpose(0, 1)
    # print(audio.min(), audio.max())
    if mono:
        audio = audio.mean(dim=0, keepdim=True)

    audio = resample_fn(audio)

    return audio


def get_statistics(args, mus):
    scaler = StandardScaler()
    pbar = tqdm.tqdm(range(len(mus.tracks)))

    spec_fn = torchaudio.transforms.Spectrogram(
        n_fft=args.model_args['n_fft'],
        win_length=args.model_args['win_length'],
        hop_length=args.model_args['hop_length'],
        power=1,
        center=True
    ).to(args.device)

    for ind in pbar:
        x = mus.tracks[ind].audio.T
        audio = torch.as_tensor(x, dtype=torch.float32).to(args.device)
        audio = audio.mean(dim=0)
        target_spec = spec_fn(audio).cpu().numpy().T
        pbar.set_description("Compute dataset statistics")
        scaler.partial_fit(np.squeeze(target_spec))

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std


def plot_spec(spectrogram, name):
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.savefig(name)
    plt.clf()
    plt.close()


def get_datetime_ref(args):
    if args.debug:
        return "0000-00-00--00-00"
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")


def delete_older_checkpoints(directory, keep=5):
    files = list(glob.glob(directory + '/*.pt'))
    files = [f for f in files if 'last' not in f and 'best' not in f]
    sorted_checkpoints = sorted(files, key=os.path.getctime, reverse=True)[keep:]
    for f in sorted_checkpoints:
        if 'best' in f:
            continue
        os.remove(f)


def save_checkpoint(iteration, model, optimizer,
                    scaler, epoch, config, output_dir, model_name, local_rank, distributed_run, is_best=False):
    if local_rank == 0:
        if is_best:
            checkpoint = {
                'iteration': iteration,
                'epoch': epoch,
                'config': config,
                'state_dict': model.state_dict() if not distributed_run else model.module.state_dict()
            }
            checkpoint_filename = "checkpoint_{}_best.pt".format(model_name, iteration)
            checkpoint_path = os.path.join(output_dir, checkpoint_filename)
            print("Saving model and optimizer state at iteration {} to {}".format(
                iteration, checkpoint_path))

            torch.save(checkpoint, checkpoint_path)
            return

        checkpoint = {
            'iteration': iteration,
            'epoch': epoch,
            'config': config,
            'state_dict': model.state_dict() if not distributed_run else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, iteration)
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print("Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path))

        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(
            output_dir, "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(os.path.abspath(symlink_src), os.path.abspath(symlink_dst))
        delete_older_checkpoints(output_dir)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(filepath, model, optimizer, scaler, epoch):
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    config = checkpoint['config']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    iteration = checkpoint['iteration'] + 1
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scaler, epoch, iteration, config