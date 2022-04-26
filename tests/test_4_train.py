import subprocess
import tempfile

import yaml


def test_train_demucs():
    with open('configs/demucs/demucs_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data_loader_args']['train']['data_root'] = 'data/MUSDB18-sample-wav'
    cfg['data_loader_args']['train']['song_lists'] = ['filelists/musdb-sample/train.txt']
    cfg['data_loader_args']['validation']['data_root'] = 'data/MUSDB18-sample-wav'
    cfg['data_loader_args']['validation']['song_lists'] = ['filelists/musdb-sample/validation.txt']
    cfg['model_args']['encoder_params']['encoder_init_channels'] = 8
    cfg['model_args']['decoder_params']['decoder_channels'] = 8
    cfg['use_stats'] = False
    cfg['num_workers'] = 1
    cfg['batch_size'] = 1
    cfg['amp'] = False
    outfile = tempfile.NamedTemporaryFile(suffix=".yaml")
    with open(outfile.name, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    subprocess.check_call(
        [
            'python',
            'train.py',
            '--config-file',
            outfile.name,
            '--debug'
        ]
    )


def test_train_d3net():
    with open('configs/d3net/vocals.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data_loader_args']['train']['data_root'] = 'data/MUSDB18-sample-wav'
    cfg['data_loader_args']['train']['song_lists'] = ['filelists/musdb-sample/train.txt']
    cfg['data_loader_args']['validation']['data_root'] = 'data/MUSDB18-sample-wav'
    cfg['data_loader_args']['validation']['song_lists'] = ['filelists/musdb-sample/validation.txt']
    cfg['use_stats'] = False
    cfg['num_workers'] = 1
    cfg['batch_size'] = 1
    cfg['amp'] = False
    outfile = tempfile.NamedTemporaryFile(suffix=".yaml")
    with open(outfile.name, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    subprocess.check_call(
        [
            'python',
            'train.py',
            '--config-file',
            outfile.name,
            '--debug'
        ]
    )
