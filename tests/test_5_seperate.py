import subprocess
import tempfile

import yaml


def test_separate_demucs():
    with open('configs/demucs/eval.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['model_path'] = 'exp/Demucs/debug/checkpoint_Demucs_best.pt'
    cfg['musdb_config']['root'] = 'data/MUSDB18-sample-wav'
    outfile = tempfile.NamedTemporaryFile(suffix=".yaml")
    with open(outfile.name, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    subprocess.check_call(
        [
            'python',
            'separate_and_evaluate.py',
            '--config',
            outfile.name,
            '--debug'
        ]
    )


def test_separate_d3net():

    with open('configs/d3net/eval.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['model_path']['vocals'] = 'exp/D3Net/debug/checkpoint_D3Net_best.pt'
    cfg['musdb_config']['root'] = 'data/MUSDB18-sample-wav'
    outfile = tempfile.NamedTemporaryFile(suffix=".yaml")
    with open(outfile.name, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    subprocess.check_call(
        [
            'python',
            'separate_and_evaluate.py',
            '--config',
            outfile.name,
            '--debug'
        ]
    )
