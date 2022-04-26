import math
import os
import random

import torch
import torch.nn.functional as F
from torch.nn import Identity
from torch.utils.data import Dataset

import torchaudio

from datasets.data_augmentations import Augmenter, pitch_shift_and_time_stretch


def load_audio(audio_path, start=0, duration=None, target_sr=44100, mono=False, cache=None):
    """
    Utility function to load audio with required sampling rate and number of channels

    :param audio_path: path to audio file, string or pathlike
    :param start: start time in seconds, defaults to beginning, t=0
    :param duration: duration to read audio, defaults to None, read entire file.
                     Pads the audio to duration if specified and actual duration is shorter
    :param target_sr: target sampling rate, audio will be resampled if it
                      differs from native sampling rate (Default:44100Hz)
    :param mono: whether to down-mix the stereo audio into mono (Default:False)
    :param cache: read audio files to RAM, use with caution, might exhaust your RAM

    :return: audio tensor of shape (CxT), where C=1 if mono else C is the number of channels in the audio
    """
    audio_path = os.path.abspath(audio_path)
    frame_offset = int(start * target_sr)
    num_frames = -1 if duration is None else int(duration * target_sr)
    if cache is not None and audio_path in cache:
        audio = cache[audio_path][:, frame_offset:frame_offset + num_frames]
    else:
        audio, native_sr = torchaudio.load(audio_path)
        # audio, native_sr = torchaudio.load(audio_path, num_frames=num_frames, frame_offset=frame_offset)
        if mono:
            audio = audio.mean(dim=0, keepdim=True)

        resample_fn = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=target_sr)
        audio = resample_fn(audio)
        if cache is not None:
            cache[audio_path] = audio

        audio = audio[:, frame_offset:frame_offset + num_frames]

    num_frames = audio.shape[1] if duration is None else int(duration * target_sr)

    pad = num_frames - audio.shape[1]
    audio = F.pad(audio, (0, pad))

    return audio, cache


def get_total_audio_length(songs):
    """
    Utility to get the total duration in seconds of all songs. Uses a .wav file in the folder for getting the duration
    Important for all the files to have the same duration (just like MUSDB)
    :param songs: list of song folders
    :return: returns the total duration in seconds of all songs in the list
    """
    total_length = 0.
    infos = {}
    for song in songs:
        # print(song)
        audio_file = [f for f in os.listdir(song) if f.endswith('.wav')][0]
        audio_file = os.path.join(song, audio_file)
        info = torchaudio.info(os.path.abspath(audio_file))
        infos[song] = info
        total_length += info.num_frames / info.sample_rate
    return total_length, infos


class SeparationDataset(Dataset):
    """
    Class to load a separation dataset. Requires data to be either MUSDB or formatted like MUSDB
    Each song has its own folder with 4 sources (bass, drums, others, vocals) and 1 mixture
    The MUSDB dataset needs to be prepared using prepare_musdb.sh
    """

    def __init__(self,
                 target_sources,
                 mixing_sources,
                 data_root,
                 song_lists,
                 sample_rate,
                 mono,
                 seq_dur,
                 augmentations,
                 pitch_shift_time_stretch_params,
                 random_mix,
                 seed,
                 iseval,
                 cache=False,
                 samples_per_epoch=None):
        """
        Initialize the dataset

        :param target_sources: The sources to be extracted. Source name string: 'vocals' or 'drums'
        :param mixing_sources: The sources to mix, must be a superset of target_sources ['vocals', 'drums', 'bass']
        :param data_root: Root of the datasets
        :param song_lists: List of song folders, folder path relative to data_root
        :param sample_rate: Target sampling rate for the audio data
        :param mono: whether to use mono or stereo
        :param iseval: Whether running in train mode or eval mode, in eval mode, following params are not used
        :param seq_dur: Audio length in seconds used for training
        :param augmentations: list of supported augmentation function names ['random_gain', 'swap_channel'].
        :param random_mix: Whether to create mixture from random songs or use the mixture of same song
        :param seed: seed for random number generators
        """
        self.target_sources = target_sources
        self.mixing_sources = mixing_sources
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.mono = mono
        self.eval = iseval
        # self.seq_dur = seq_dur if not iseval else None
        self.seq_dur = seq_dur
        self.augment = Augmenter(augmentations) if not iseval else Identity()
        self.pitch_shift_time_stretch_params = pitch_shift_time_stretch_params
        self.random_mix = random_mix if not iseval else False
        self.seed = seed
        self.songs = []
        for song_list in song_lists:
            with open(song_list, 'r') as f:
                songs = f.readlines()
            # print(song_list)
            self.songs += [os.path.join(self.data_root, s.strip()) for s in songs if s != '\n']
        self.total_audio_length, self.song_infos = get_total_audio_length(self.songs)
        self.cache = None
        self.samples_per_epoch = samples_per_epoch
        if cache:
            self.cache = {}
        # print(self.songs)

    def __len__(self):
        """
        Function to return the length of the dataset, used for deciding number of iterations per epoch
        :return: returnt the number of segments of length seq_dur if training, else return the number of songs
        """
        if self.eval:
            return len(self.songs)
        if self.samples_per_epoch:
            return self.samples_per_epoch
        else:
            length = int(math.ceil(self.total_audio_length - len(self.songs) * self.seq_dur))
            if length > 0:
                return length
            else:
                return len(self.songs)

    def __getitem__(self, index):
        """
        Main logic to read the audio files and create the mixture signal and target signal
        :param index: song index during eval, but unused during training, as we sample randomly
        :return: mixture (CxT) and target (CxT), where C is number of channels and T is time
        """
        sources = {}

        if self.eval:
            song = self.songs[index]
            start = 0
            if self.seq_dur is not None:
                info = self.song_infos[song]
                duration = info.num_frames / info.sample_rate
                start = (duration / 2) - (self.seq_dur / 2)
            sources = {f: [start, self.seq_dur, os.path.join(song, f + '.wav'),
                           self.song_infos[song]] for f in self.mixing_sources}
        else:
            if self.random_mix:
                for source in self.mixing_sources:
                    song = random.choice(self.songs)
                    info = self.song_infos[song]
                    duration = info.num_frames / info.sample_rate
                    end = 0 if duration < self.seq_dur else duration - self.seq_dur
                    random_start = random.uniform(0, end)
                    sources[source] = [random_start, self.seq_dur, os.path.join(song, source + '.wav'), info]
            else:
                song = random.choice(self.songs)
                info = self.song_infos[song]
                duration = info.num_frames / info.sample_rate
                end = 0 if duration < self.seq_dur else duration - self.seq_dur
                random_start = random.uniform(0, end)
                sources = {f: [random_start, self.seq_dur, os.path.join(song, f + '.wav'),
                               info] for f in self.mixing_sources}
        source_audios = []
        target_audios = []
        for source in sources:
            start, duration, audio_path, info = sources[source]
            audio, self.cache = load_audio(audio_path, start=start, duration=duration,
                                           target_sr=self.sample_rate, mono=self.mono, cache=self.cache)
            if not self.eval:
                audio = self.augment(audio)
                if self.pitch_shift_time_stretch_params is not None:
                    # apply pitch shift and time stretch separately
                    out_length = int((1 - 0.01 * self.pitch_shift_time_stretch_params['max_tempo']) * audio.shape[-1])
                    audio = pitch_shift_and_time_stretch(audio, source, self.sample_rate,
                                                         **self.pitch_shift_time_stretch_params)
                    audio = audio[:, :out_length]
            source_audios.append(audio)
            if source in self.target_sources:
                target_audios.append(audio)

        mixture = sum(source_audios)
        target = torch.stack(target_audios, dim=0)

        return mixture, target
