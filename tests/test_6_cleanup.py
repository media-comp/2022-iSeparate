import os
import shutil


def test_cleanup():
    if os.path.exists('exp/Demucs/debug'):
        shutil.rmtree('exp/Demucs/debug')
    if os.path.exists('exp/D3Net/debug'):
        shutil.rmtree('exp/D3Net/debug')
    if os.path.exists('data/MUSDB18-sample'):
        shutil.rmtree('data/MUSDB18-sample')
    # if os.path.exists('data/MUSDB18-sample-wav'):
    #     shutil.rmtree('data/MUSDB18-sample-wav')