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
import zipfile
import tarfile
from pathlib import Path
import glob
import random

import stempeg
import tqdm
import wget

from load_split_music_data import splitMusicData
from load_split_speech_data import splitSpeechData
from load_split_noise_data import splitNoiseData
from mix_data import mixData

def download(root, keep_wavs_only=True):
    """Download the MTASS data"""
    """MTASS consists of 3 different datasets. So download 3 datasets."""
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
        
    aishell_url = "https://us.openslr.org/resources/33/data_aishell.tgz"
    aishell_filename = "aishell.tgz"
    
    print("Downloading {} Dataset to {}...".format(aishell_filename, root))
    
    # data = urlopen(url)
    file_path = os.path.join(root, aishell_filename)
    if not os.path.exists(file_path):
        wget.download(aishell_url, file_path)
    tar_ref = tarfile.open(file_path, "r")
    tar_ref.extractall(os.path.join(root))
    tar_ref.close()
    if keep_wavs_only:
        os.unlink(file_path)
    tar_list=glob.glob(os.path.join(root,"data_aishell","wav","*.tar.gz"))
    for tar_path in tar_list:
        print("extract "+tar_path)
        tar_ref = tarfile.open(tar_path, "r")
        tar_ref.extractall(os.path.join(root,"data_aishell","wav"))
        tar_ref.close()
        if keep_wavs_only:
            os.unlink(tar_path)
        
    print(" Download and extraction of {} complete!".format(aishell_filename))
    

    dsd_url = "http://liutkus.net/DSD100.zip"
    dsd_filename = "dsd100.zip"
    
    print("Downloading {} Dataset to {}...".format(dsd_filename, root))
    
    # data = urlopen(url)
    file_path = os.path.join(root, dsd_filename)
    if not os.path.exists(file_path):
        wget.download(dsd_url, file_path)
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall(os.path.join(root))
    zip_ref.close()
    if keep_wavs_only:
        os.unlink(file_path)
        
    print(" Download and extraction of {} complete!".format(dsd_filename))
   
    
    os.system('cd '+root+'; bash download-dns-challenge-3.sh')
    file_path = os.path.join(root, "datasets/datasets.noise.tar.bz2")
    tar_ref = tarfile.open(file_path)
    tar_ref.extractall(file_path)
    tar_ref.close()
    if keep_wavs_only:
        os.unlink(file_path)
    print(" Download and extraction of {} complete!".format(file_path))


def prepare_mtass(
    root_path="data/MTASS",
    wav_output_path="data/MTASS-wav",
    num_file=1000,
    keep_wavs_only=True,
    filelists_dir="filelists/MTASS",
    make_symlink=True,
):
    """
    download(root_path, keep_wavs_only=keep_wavs_only)
    splitMusicData(os.path.join(root_path, "DSD100/Mixtures"), os.path.join(wav_output_path, "music_data/"), "train")
    splitMusicData(os.path.join(root_path, "DSD100/Mixtures"), os.path.join(wav_output_path, "music_data/"), "test")
    splitSpeechData(os.path.join(root_path, "data_aishell/wav"), os.path.join(wav_output_path, "speech_data/"), "train")
    splitSpeechData(os.path.join(root_path, "data_aishell/wav"), os.path.join(wav_output_path, "speech_data/"), "test")
    
    #split noise data to train and test
    noise_train_path = os.path.join(root_path, "datasets/datasets/noise/train")
    noise_test_path = os.path.join(root_path, "datasets/datasets/noise/test")
    if not os.path.exists(noise_train_path):
        os.makedirs(noise_train_path)
    if not os.path.exists(noise_test_path):
        os.makedirs(noise_test_path)
    count = 0
    for p in glob.glob(os.path.join(root_path, "datasets/datasets/noise/**.wav")):
        if count % 10 == 0:
            shutil.move(p, noise_test_path)
        else:
            shutil.move(p, noise_train_path)
        count += 1
    """
    splitNoiseData(os.path.join(root_path, "datasets/datasets/noise"), os.path.join(wav_output_path, "noise_data/"), "train")
    splitNoiseData(os.path.join(root_path, "datasets/datasets/noise"), os.path.join(wav_output_path, "noise_data/"), "test")        
    
    #mix data
    mixData(wav_output_path, wav_output_path, type="train", num_file=num_file//10*9)
    mixData(wav_output_path, wav_output_path, type="test", num_file=num_file//10)

    if wav_output_path != "data/MTASS" and make_symlink:
        os.makedirs("data", exist_ok=True)
        os.symlink(
            os.path.abspath(wav_output_path),
            os.path.abspath("data/MTASS-wav"),
            target_is_directory=True,
        )

    print("Creating Train/Validation/Test filelists...")

    train_songs = ["train/" + f for f in os.listdir(os.path.join(wav_output_path, "train"))]
    test_songs = ["test/" + f for f in os.listdir(os.path.join(wav_output_path, "test"))]
    validation_songs = random.sample(train_songs, len(train_songs)//10)

    train_songs = [f for f in train_songs if f not in validation_songs]

    os.makedirs(filelists_dir, exist_ok=True)

    with open(os.path.join(filelists_dir, "train.txt"), "w") as f:
        for song in train_songs:
            f.write(song + "\n")
    with open(os.path.join(filelists_dir, "test.txt"), "w") as f:
        for song in test_songs:
            f.write(song + "\n")
    with open(os.path.join(filelists_dir, "validation.txt"), "w") as f:
        for song in validation_songs:
            f.write(song + "\n")

    print("MTASS dataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/MTASS",
        help="path to store the stems version of the MTASS dataset",
    )
    parser.add_argument(
        "--wav-root",
        type=str,
        default="data/MTASS-wav",
        help="path to store the wav version of the MUSDB dataset",
    )
    parser.add_argument(
        "--num_file",
        type=int,
        default=1000,
        help="total number of files in MTASS dataset",
    )
    parser.add_argument(
        "--filelists-dir",
        type=str,
        default="filelists/MTASS",
        help="path to store the wav version of the MTASS dataset",
    )
    parser.add_argument(
        "--keep-wav-only",
        action="store_true",
        help="keep only the wav files and delete intermediate files",
    )
    parser.add_argument(
        "--make-symlink",
        action="store_true",
        help="create symlink in project directory if the dataset is in another directory",
    )

    args = parser.parse_args()
    prepare_mtass(
        root_path=args.root,
        wav_output_path=args.wav_root,
        num_file=args.num_file,
        keep_wavs_only=args.keep_wav_only,
        filelists_dir=args.filelists_dir,
        make_symlink=args.make_symlink,
    )
