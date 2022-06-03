# Some parts of the code are borrowed from the official Demucs code
# https://github.com/facebookresearch/demucs/blob/v2/demucs/
# with the following copyright notice:
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the MIT license

import math
import random

import julius
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py
def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


# https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py
def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def get_nonlinearity(nonlinearity="glu"):
    if nonlinearity == "glu":
        return nn.GLU(dim=1)
    elif nonlinearity == "relu":
        return nn.ReLU()
    else:
        return nn.Identity()


# https://github.com/facebookresearch/demucs/blob/v2/demucs/utils.py
def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        encoder_init_channels,
        encoder_num_layers,
        encoder_nonlinearity,
        encoder_kernel_size,
        encoder_stride,
        encoder_growth_rate,
    ):
        super(Encoder, self).__init__()
        self.num_layers = encoder_num_layers
        self.kernel_size = encoder_kernel_size
        self.stride = encoder_stride
        in_channels = input_channels
        out_channels = encoder_init_channels
        activation = get_nonlinearity(encoder_nonlinearity)
        self.encoder_layers = nn.ModuleList()
        for i in range(encoder_num_layers):
            enc_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, encoder_kernel_size, encoder_stride),
                nn.ReLU(),
                nn.Conv1d(out_channels, int(encoder_growth_rate * out_channels), 1),
                activation,
            )

            in_channels = out_channels
            out_channels = int(encoder_growth_rate * out_channels)
            self.encoder_layers.append(enc_layer)

        self.out_channels = in_channels

    def forward(self, x):

        outputs = []
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
            outputs.append(x)

        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        num_sources,
        audio_channels,
        decoder_channels,
        decoder_kernel_size,
        decoder_stride,
        decoder_num_layers,
        decoder_growth_rate,
        decoder_nonlinearity,
        decoder_context_size,
    ):
        super(Decoder, self).__init__()
        self.num_layers = decoder_num_layers
        self.kernel_size = decoder_kernel_size
        self.stride = decoder_stride
        self.num_sources = num_sources
        in_channels = decoder_channels
        out_channels = num_sources * audio_channels
        activation = get_nonlinearity(decoder_nonlinearity)
        self.decoder_layers = nn.ModuleList()
        for i in range(decoder_num_layers):
            dec_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels, int(decoder_growth_rate * in_channels), decoder_context_size
                ),
                activation,
                nn.ConvTranspose1d(in_channels, out_channels, decoder_kernel_size, decoder_stride),
                nn.ReLU() if i > 0 else nn.Identity(),
            )

            out_channels = in_channels
            in_channels = int(decoder_growth_rate * in_channels)
            self.decoder_layers.insert(0, dec_layer)

    def forward(self, x, skips):
        for i, dec_layer in enumerate(self.decoder_layers):
            x = x + center_trim(skips[-i - 1], x)
            x = dec_layer(x)
        return x


class Demucs(nn.Module):
    def __init__(
        self,
        audio_channels,
        input_channels,
        num_sources,
        encoder_params,
        decoder_params,
        blstm_hidden_dim,
        blstm_num_layers,
        rescale,
        resample,
    ):
        super(Demucs, self).__init__()
        self.num_sources = num_sources
        self.audio_channels = audio_channels
        self.resample = resample
        input_channels = audio_channels if input_channels is None else input_channels
        self.encoder = Encoder(input_channels=input_channels, **encoder_params)
        blstm_hidden_dim = (
            self.encoder.out_channels if blstm_hidden_dim is None else blstm_hidden_dim
        )
        self.center_blstm = nn.LSTM(
            bidirectional=True,
            num_layers=blstm_num_layers,
            hidden_size=blstm_hidden_dim,
            input_size=self.encoder.out_channels,
        )
        self.center_projection = nn.Linear(2 * blstm_hidden_dim, self.encoder.out_channels)
        self.decoder = Decoder(
            num_sources=num_sources, audio_channels=audio_channels, **decoder_params
        )
        rescale_module(self, reference=rescale)

    def valid_length(self, length):
        if self.resample:
            length *= 2

        for _ in range(self.encoder.num_layers):
            length = math.ceil((length - self.encoder.kernel_size) / self.encoder.stride) + 1
            length = max(1, length)

        for idx in range(self.decoder.num_layers):
            length = (length - 1) * self.decoder.stride + self.decoder.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mixture, target=None, **kwargs):
        x = mixture

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)

        x = (x - mean) / (1e-5 + std)

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        enc_outs = self.encoder(x)

        x = enc_outs[-1]
        x = self.center_blstm(x.permute(2, 0, 1))[0]
        x = self.center_projection(x).permute(1, 2, 0)

        x = self.decoder(x, enc_outs)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)

        x = x * std + mean
        x = x.view(x.shape[0], self.num_sources, self.audio_channels, x.shape[-1])

        outputs = (x,)
        if target is not None:
            target = center_trim(target, x)
            outputs += (target,)

        return outputs

    def separate(self, x, shifts=0, sample_rate=44100):
        orig_length = x.shape[-1]
        if shifts > 0:
            max_shift = int(0.5 * sample_rate)
            out = 0
            for _ in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = torch.roll(x, shifts=offset, dims=-1)
                shifted[..., :offset] = 0
                valid_length = self.valid_length(shifted.shape[-1])
                pad = valid_length // 2
                shifted = F.pad(shifted, (pad, pad))
                shifted_out = self.forward(shifted)[0]
                shifted_out = center_trim(shifted_out, valid_length)
                out += shifted_out[..., max_shift - offset :]
            out /= shifts
        else:
            valid_length = self.valid_length(orig_length)
            pad = valid_length // 2
            x = F.pad(x, (pad, pad))
            out = self.forward(x)[0]

        out = center_trim(out, orig_length)
        return out
