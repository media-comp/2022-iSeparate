import argparse
import os
import shutil
import sys
import functools
import multiprocessing

import musdb
import museval
import torch
import torchaudio
import tqdm
import wget
import yaml

from models.model_switcher import get_separation_funcs
from utils.common_utils import preprocess_audio


def check_n_download_models(model_path):
    base_url = 'https://huggingface.co/subatomicseer/D3Net/resolve/main/'
    for target in model_path.keys():
        if model_path[target] is not None:
            if not os.path.exists(model_path[target]):
                print('Downloading {} model...'.format(target))
                file_path = model_path[target]
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                url = base_url+'{}.pt'.format(target)
                wget.download(url, file_path)


def separate(audio_file_path, eval_config_path, outdir='separation_results'):
    with open(eval_config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # TODO: Currently only D3Net model is available for download
    if cfg['model_name'] != 'D3Net':
        print('Currently only D3Net model is available for download,'
              ' Skipping inference with {}...'.format(cfg['model_name']))
    check_n_download_models(cfg['model_path'])
    model_loader, separator = get_separation_funcs(cfg['model_name'])
    model = model_loader(cfg['model_path'])

    audio, sr = torchaudio.load(audio_file_path)
    audio = preprocess_audio(audio, sr, model)
    model_sr = model[list(model.keys())[0]]['config'].data_loader_args['train']['sample_rate']
    estimates = separator(
        model,
        audio,
        **cfg['eval_params']
    )

    song_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    outdir = os.path.join(outdir, song_name)
    os.makedirs(outdir, exist_ok=True)

    for key in estimates:
        output = estimates[key].cpu()
        outfilepath = os.path.join(outdir, '{}.wav'.format(key))
        torchaudio.save(outfilepath, output, model_sr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path to the evaluation config file")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="path to the audio file to be separated")
    parser.add_argument('-o', '--outdir', type=str, default='separation_results',
                        help="directory for saving the results, will create a folder with the input filename")

    args = parser.parse_args()

    separate(args.input, args.config, args.outdir)
