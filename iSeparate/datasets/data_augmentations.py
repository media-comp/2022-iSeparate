import io
import random
import subprocess
import tempfile
from shutil import which

import librosa
import numpy as np
import torch
from scipy.io import wavfile


def i16_pcm(wav):
    if wav.dtype == np.int16:
        return wav
    return (wav * 2**15).clamp_(-(2**15), 2**15 - 1).short()


def f32_pcm(wav):
    if wav.dtype == np.float32:
        return wav
    return wav.float() / 2**15


def random_gain(audio, low=0.25, high=1.25):
    """
    Utility function to apply a random gain between low and high to the audio.

    :param audio: torch tensor of shape CxT
    :param low: minimum gain to be applied, float (Default:0.25)
    :param high: maximum gain to be applied, float (Default:1.25)
    :return: returns the audio after gain is applied, torch tensor of shape CxT
    """

    gain = low + random.random() * (high - low)
    return audio * gain


def swap_channel(audio, p=0.5):
    """
    Utility function to randomly swap the left and right channels in case of stereo inputs

    :param audio: torch tensor of chape CxT
    :param p: probability of swapping (default: 0.5)

    :return: audio with channels swapped with probability p
    """
    swap = random.random() < p
    if audio.shape[0] == 1 or not swap:
        return audio
    else:
        return torch.flip(audio, [0])


def sign_flip(audio):
    signs = torch.randint(2, (1, 1), device=audio.device, dtype=torch.float32)
    audio = audio * (2 * signs - 1)
    return audio


def pitch_shift_and_time_stretch(
    audio,
    source,
    sample_rate=44100,
    max_pitch=2,
    max_tempo=12,
    tempo_std=5,
    p=0.2,
    quick=True,
    use_soundstretch=False,
):
    if random.random() < p:
        delta_pitch = random.randint(-max_pitch, max_pitch)
        delta_tempo = random.gauss(0, tempo_std)
        delta_tempo = min(max(-max_tempo, delta_tempo), max_tempo)

        if which("soundstretch") is not None and use_soundstretch:
            outfile = tempfile.NamedTemporaryFile(suffix=".wav")
            in_ = io.BytesIO()
            wavfile.write(in_, sample_rate, i16_pcm(audio).t().numpy())
            command = [
                "soundstretch",
                "stdin",
                outfile.name,
                f"-pitch={delta_pitch}",
                f"-tempo={delta_tempo:.6f}",
            ]
            if quick:
                command += ["-quick"]
            if source == "vocals":
                command += ["-speech"]
            try:
                subprocess.run(command, capture_output=True, input=in_.getvalue(), check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
            sr, audio = wavfile.read(outfile.name)
            audio = audio.copy()
            audio = f32_pcm(torch.from_numpy(audio).t())
            assert sr == sample_rate
        else:
            # fall back to librosa if soundstretch is not installed
            audio = audio.numpy()
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=float(delta_pitch))
            tempo = 1.0 + delta_tempo * 0.01
            audio = librosa.effects.time_stretch(audio, rate=tempo)
            audio = torch.from_numpy(audio)
    return audio


class Augmenter:
    """
    Class to combine and apply data augmentations
    """

    def __init__(self, augmentation_list):
        """
        initialize the required augmentations
        :param augmentation_list: list of strings with the function names.
        E.g. ['swap_channel', 'random_gain']
        """
        self.augmentations = [globals()[aug] for aug in augmentation_list]

    def __call__(self, audio):
        """

        :param audio: audio to be augmented, shape CxT
        :return: augmented audio of same shape as input
        """
        for aug_fn in self.augmentations:
            audio = aug_fn(audio)

        return audio
