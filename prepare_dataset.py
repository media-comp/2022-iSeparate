"""
Most of this code is borrowed (with some modifications) from https://github.com/sigsep/sigsep-mus-db
which was released under the following license:

The MIT License (MIT)

Copyright (c) 2015, 2016, 2017 Fabian-Robert Stöter
Copyright (c) 2018, 2019 Inria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import errno
import os
import shutil
import wget
import zipfile

import stempeg
import tqdm
from musdb import DB
from pathlib import Path


def download(root, download_sample=False, keep_wavs_only=True):
    """Download the MUSDB data"""
    if os.path.exists(os.path.join(root, "train")):
        return

    # download files
    try:
        os.makedirs(os.path.join(root))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    if download_sample:
        url = 'https://zenodo.org/record/3270814/files/MUSDB18-7-STEMS.zip'
        filename = 'MUSDB18-7-STEMS.zip'
    else:
        url = 'https://zenodo.org/record/1117372/files/musdb18.zip'
        filename = 'MUSDB18.zip'

    print('Downloading {} Dataset to {}...'.format(filename, root))
    # data = urlopen(url)
    file_path = os.path.join(root, filename)
    if not os.path.exists(file_path):
        wget.download(url, file_path)
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(os.path.join(root))
    zip_ref.close()
    if keep_wavs_only:
        os.unlink(file_path)

    print(' Download and extraction complete!')


def convert_to_wav(musdb_root, output_root, extension='.wav', keep_wavs_only=True):

    mus = DB(root=musdb_root, download=False)

    print('converting MUSDB dataset to wav form...')
    for track in tqdm.tqdm(mus):

        track_estimate_dir = Path(
            output_root, track.subset, track.name
        )
        track_estimate_dir.mkdir(exist_ok=True, parents=True)
        # write out tracks to disk

        stempeg.write_audio(
            path=str(track_estimate_dir / Path('mixture').with_suffix(extension)),
            data=track.audio,
            sample_rate=track.rate
        )
        for name, track in track.targets.items():
            stempeg.write_audio(
                path=str(track_estimate_dir / Path(name).with_suffix(extension)),
                data=track.audio,
                sample_rate=track.rate
            )
    if keep_wavs_only:
        shutil.rmtree(musdb_root)


def prepare_musdb(root_path='data/MUSDB18-7s', wav_output_path='data/MUSDB18-wav',
                  download_sample=True, keep_wavs_only=True, filelists_dir='filelists/musdb', make_symlink=True):
    if not os.path.exists(os.path.join(wav_output_path, "train")):
        download(root_path, download_sample, keep_wavs_only=keep_wavs_only)
        convert_to_wav(root_path, wav_output_path, keep_wavs_only=keep_wavs_only)

    if wav_output_path != 'data/MUSDB18-wav' and make_symlink:
        os.makedirs('data', exist_ok=True)
        os.symlink(os.path.abspath(wav_output_path), os.path.abspath('data/MUSDB18-wav'), target_is_directory=True)

    print('Creating Train/Validation/Test filelists...')

    mus = DB(root=wav_output_path, is_wav=True)

    train_songs = ['train/' + f for f in os.listdir(os.path.join(wav_output_path, 'train'))]
    test_songs = ['test/' + f for f in os.listdir(os.path.join(wav_output_path, 'test'))]
    validation_songs = ['train/' + f for f in mus.setup['validation_tracks']]

    train_songs = [f for f in train_songs if f not in validation_songs]

    os.makedirs(filelists_dir, exist_ok=True)

    with open(os.path.join(filelists_dir, 'train.txt'), 'w') as f:
        for song in train_songs:
            f.write(song + '\n')
    with open(os.path.join(filelists_dir, 'test.txt'), 'w') as f:
        for song in test_songs:
            f.write(song + '\n')
    with open(os.path.join(filelists_dir, 'validation.txt'), 'w') as f:
        for song in validation_songs:
            f.write(song + '\n')

    print('MUSDB dataset preparation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/MUSDB18-7s',
                        help="path to store the stems version of the MUSDB dataset")
    parser.add_argument('--wav-root', type=str, default='data/MUSDB18-wav',
                        help="path to store the wav version of the MUSDB dataset")
    parser.add_argument('--filelists-dir', type=str, default='filelists/musdb',
                        help="path to store the wav version of the MUSDB dataset")
    parser.add_argument('--download-sample', action='store_true',
                        help="download only a 7s sample dataset for testing")
    parser.add_argument('--keep-wav-only', action='store_true',
                        help="keep only the wav files and delete intermediate files")
    parser.add_argument('--make-symlink', action='store_true',
                        help="create symlink in project directory if the dataset is in another directory")

    args = parser.parse_args()
    prepare_musdb(root_path=args.root, wav_output_path=args.wav_root,
                  download_sample=args.download_sample, keep_wavs_only=args.keep_wav_only,
                  filelists_dir=args.filelists_dir, make_symlink=args.make_symlink)
