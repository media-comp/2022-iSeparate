import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn import Parameter


class BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(BNReLUConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation,
                              padding=padding)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class D2Block(nn.Module):
    """
    Implementation of the multi dilated convolution from Eq. 2 in the paper
    """

    def __init__(self, in_channels, growth_rate, kernel_size, padding, num_layers, num_outs):
        super(D2Block, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.num_outs = num_outs
        self.splits = [growth_rate] * num_layers
        self.init_conv = BNReLUConv(in_channels=in_channels,
                                    out_channels=growth_rate * num_layers,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilation=1,
                                    padding=padding)
        in_channels = growth_rate * num_outs

        if self.num_layers > 1:
            self.convs = nn.ModuleList()
            for i in range(num_layers - 1):
                dilation = 2 ** (i + 1)
                out_channels = growth_rate * (num_layers - i - 1)
                self.convs.append(
                    BNReLUConv(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, dilation=dilation,
                               padding=dilation)
                )
                in_channels = growth_rate * num_outs

    def forward(self, x):
        out = self.init_conv(x)
        if self.num_layers > 1:
            tensor_splits = [t.clone() for t in torch.split(out, split_size_or_sections=self.splits, dim=1)]
            for i, (split, conv) in enumerate(zip(tensor_splits, self.convs)):
                out = conv(split)
                for j in range(self.num_layers - i - 1):
                    tensor_splits[j + 1 + i] += out[:, j * self.growth_rate:(j + 1) * self.growth_rate]
            out = torch.cat(tensor_splits, dim=1)
        return out[:, -self.num_outs * self.growth_rate:]


class D3block(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, num_blocks, kernel_size, padding, num_outs):
        super(D3block, self).__init__()

        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_outs = num_outs
        self.splits = [growth_rate] * num_blocks
        self.init_block = D2Block(in_channels, growth_rate * num_blocks, kernel_size, padding, num_layers, num_outs)

        in_channels = growth_rate * num_outs

        if self.num_blocks > 1:
            self.blocks = nn.ModuleList()
            for i in range(num_blocks - 1):
                out_channels = growth_rate * (num_blocks - i - 1)
                self.blocks.append(
                    D2Block(in_channels=in_channels, growth_rate=out_channels,
                            kernel_size=kernel_size, padding=padding, num_layers=num_layers, num_outs=num_outs)
                )
                in_channels = growth_rate * num_outs

    def forward(self, x):
        out = self.init_block(x)
        if self.num_blocks > 1:
            tensor_splits = [t.clone() for t in torch.split(out, split_size_or_sections=self.splits, dim=1)]
            for i, (split, block) in enumerate(zip(tensor_splits, self.blocks)):
                out = block(split)
                for j in range(self.num_blocks - i - 1):
                    tensor_splits[j + 1 + i] += out[:, j * self.growth_rate:(j + 1) * self.growth_rate]
            out = torch.cat(tensor_splits, dim=1)
        return out[:, -self.num_outs * self.growth_rate:]


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.tconv(self.bn(x))


class Downsample(nn.Module):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2)):
        super(Downsample, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)


class MD3Net(nn.Module):
    def __init__(self, in_channels, growth_rates, num_layers, num_blocks, kernel_size, padding, num_outs):
        super(MD3Net, self).__init__()
        assert len(growth_rates) == len(num_layers)
        assert len(num_layers) % 2 != 0
        num_downsamples = num_upsamples = (len(num_layers) - 1) // 2

        # downsamples
        self.downsample_blocks = nn.ModuleList()
        downsample_output_channels = []
        for k, n_layers, n_blocks in zip(growth_rates[:num_downsamples],
                                         num_layers[:num_downsamples], num_blocks[:num_downsamples]):
            self.downsample_blocks.append(D3block(in_channels, k, n_layers, n_blocks, kernel_size, padding, num_outs))
            in_channels = num_outs * k
            downsample_output_channels.append(in_channels)

        downsample_output_channels = downsample_output_channels[::-1]

        # bottleneck
        self.bottleneck_block = D3block(in_channels, growth_rates[num_downsamples], num_layers[num_downsamples],
                                        num_blocks[num_downsamples], kernel_size, padding, num_outs)

        n_channels = num_outs * growth_rates[num_downsamples]
        # upsamples
        self.upsample_blocks = nn.ModuleList()
        for i, (k, n_layers, n_blocks) in enumerate(zip(growth_rates[num_upsamples + 1:],
                                                        num_layers[num_upsamples + 1:],
                                                        num_blocks[num_upsamples + 1:])):
            n_channels_upsample = n_channels
            n_channels_concat = n_channels_upsample + downsample_output_channels[i]
            self.upsample_blocks.append(
                nn.ModuleDict(
                    {
                        'upsample': Upsample(n_channels_upsample, n_channels_upsample),
                        'D3block': D3block(n_channels_concat, k, n_layers, n_blocks, kernel_size, padding, num_outs)
                    }
                )
            )
            n_channels = num_outs * k

    def forward(self, x):
        downs = []
        out = x
        for ds_block in self.downsample_blocks:
            out = ds_block(out)
            downs.append(out)
            out = F.avg_pool2d(out, kernel_size=(2, 2), stride=(2, 2))
        # reverse the list as uplsampling is mirror image of downsampling
        downs = downs[::-1]

        out = self.bottleneck_block(out)

        # outs = []
        for i, us_block in enumerate(self.upsample_blocks):
            out = us_block['upsample'](out)
            out = torch.cat([out, downs[i]], dim=1)
            out = us_block['D3block'](out)
            # outs.append(out.clone())
        return out


class SpectrogramFunction(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super(SpectrogramFunction, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
            center=True
        )
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True
        )
        self.mel_scale_fn = torchaudio.transforms.MelScale(
            n_mels=128, sample_rate=44100, n_stft=n_fft//2+1
        )
        for p in self.parameters():
            p.requires_grad = True

    def transform(self, x, patch_length=None, return_magnitude=True):
        with torch.cuda.amp.autocast(enabled=False):
            stft = self.stft(x)
            if patch_length is not None:
                stft = stft[..., :patch_length]

            if return_magnitude:
                magnitude = stft.abs()
                return magnitude
            else:
                return stft

    def inverse_transform(self, stft=None, magnitude=None, phase=None, length=None):
        with torch.cuda.amp.autocast(enabled=False):
            if stft is not None:
                waveform = self.istft(stft, length=length)
            else:
                assert magnitude is not None and phase is not None
                stft = torch.polar(magnitude, phase)
                waveform = self.istft(stft, length=length)
            return waveform

    def mel_scale(self, stft):
        return self.mel_scale_fn(stft)


class MMD3Net(nn.Module):
    def __init__(self,
                 audio_channels,
                 init_channels,
                 band_split_idxs,
                 valid_signal_index,
                 growth_rates,
                 num_layers,
                 num_blocks,
                 kernel_size,
                 padding,
                 num_outs,
                 input_mean=None,
                 input_scale=None,
                 n_fft=4096,
                 hop_length=1024,
                 win_length=1024,
                 patch_length=256
                 ):
        super(MMD3Net, self).__init__()

        assert len(band_split_idxs) + 1 == len(growth_rates['bands']) == len(num_layers['bands']) \
               == len(num_blocks['bands']) == len(num_outs['bands']) == len(init_channels['bands'])

        max_bin = None
        if valid_signal_index is not None:
            self.band_split_idxs = band_split_idxs + [valid_signal_index]
            max_bin = valid_signal_index

        nb_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = nb_bins
        self.nb_output_bins = self.nb_bins

        self.patch_length = patch_length
        self.spectrogram_function = SpectrogramFunction(n_fft, hop_length, win_length)

        # full band network
        self.full_net = nn.Sequential(
            nn.Conv2d(audio_channels, init_channels['full'], kernel_size=kernel_size, padding=padding),
            MD3Net(init_channels['full'],
                   growth_rates=growth_rates['full'],
                   num_layers=num_layers['full'],
                   num_blocks=num_blocks['full'],
                   kernel_size=kernel_size,
                   padding=padding,
                   num_outs=num_outs['full']
                   )
        )

        # band networks
        max_final_growth_rate = max([k[-1] * n for k, n in zip(growth_rates['bands'], num_outs['bands'])])
        self.band_nets = nn.ModuleList()
        for i, (init_features, growth_rate, num_layer, num_block, num_out) in enumerate(zip(init_channels['bands'],
                                                                                            growth_rates['bands'],
                                                                                            num_layers['bands'],
                                                                                            num_blocks['bands'],
                                                                                            num_outs['bands'])):
            init_conv = nn.Conv2d(audio_channels, init_features, kernel_size=kernel_size, padding=padding)
            d3net = MD3Net(init_features, growth_rate, num_layer, num_block, kernel_size, padding, num_out)
            if max_final_growth_rate > growth_rate[-1] * num_out:
                feature_match_layer = BNReLUConv(growth_rate[-1] * num_out,
                                                 max_final_growth_rate,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 dilation=1,
                                                 stride=1)
                d3net = nn.Sequential(d3net, feature_match_layer)
            net = nn.Sequential(
                init_conv,
                d3net
            )
            self.band_nets.append(net)

        # final D2 block
        in_channels = growth_rates['full'][-1] * num_outs['full'] + max_final_growth_rate
        self.final_d2_block = D2Block(in_channels=in_channels,
                                      growth_rate=growth_rates['final'],
                                      num_layers=num_layers['final'],
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      num_outs=num_outs['final']
                                      )

        in_channels = growth_rates['final'] * num_outs['final']
        self.bn_out = nn.BatchNorm2d(in_channels)
        self.gate_out = nn.Sequential(nn.Conv2d(in_channels, audio_channels, kernel_size=(1, 1), stride=(1, 1)),
                                      nn.Sigmoid())
        self.filter_out = nn.Conv2d(in_channels, audio_channels, kernel_size=(1, 1), stride=(1, 1))

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def process_spects(self, inp):
        valid_invalid_splits = torch.tensor_split(inp, self.band_split_idxs, dim=3)
        invalid_band = valid_invalid_splits[-1]
        x = torch.cat(valid_invalid_splits[:-1], dim=3)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # iterate over the bands except the last one, because it is outside the valid range and goes straight
        # through to the output, basically to ensure symmetry of the up and down sampling
        valid_bands = torch.tensor_split(x, self.band_split_idxs[:-1], dim=3)
        band_outs = []
        for band, band_net in zip(valid_bands, self.band_nets):
            out = band_net(band)
            band_outs.append(out)

        band_out_concat = torch.cat(band_outs, dim=3)

        full_out = self.full_net(x)

        concat_all = torch.cat([band_out_concat, full_out], dim=1)

        final_d2_out = self.final_d2_block(concat_all)

        final_out = self.bn_out(final_d2_out)
        gate = self.gate_out(final_out)
        filt = self.filter_out(final_out)

        final_out = gate * filt

        # apply output scaling
        final_out *= self.output_scale
        final_out += self.output_mean

        final_out = F.relu(final_out)
        final_out = torch.cat([final_out, invalid_band], dim=3)
        return final_out

    def forward(self, x, y=None, return_input_spec=False, patch_length=-1):
        """

        :param x:
        :param y:
        :param return_input_spec:
        :param patch_length: None: use full signal, pad to avoid errors
                               -1: use self.patch_length
                          integer: use the provided value
        :return:
        """
        if patch_length == -1:
            patch_length = self.patch_length

        x = self.spectrogram_function.transform(x, patch_length=patch_length, return_magnitude=True).transpose(2, 3)

        left_pad = 0
        right_pad = 0
        if patch_length is None:
            left_pad = self.patch_length
            right_pad = 2 * self.patch_length - (x.shape[2] % self.patch_length)

        x = F.pad(x, (0, 0, left_pad, right_pad))

        if y is not None:
            assert y.shape[1] == 1, 'D3Net supports only single target training'
            # y = y.squeeze(1)
            y = self.spectrogram_function.transform(y, patch_length=patch_length,
                                                    return_magnitude=True).transpose(3, 4)
            # y = F.pad(y, (0, 0, left_pad, right_pad))

        final_out = self.process_spects(x)
        start = None if left_pad == 0 else left_pad
        end = None if right_pad == 0 else -right_pad
        final_out = final_out[:, :, start:end].unsqueeze(1)
        outputs = (final_out, )
        if y is not None:
            outputs += (y, )
        if return_input_spec:
            outputs += (x[:, :, start:end], )
        return outputs

    def separate(self, audio, hop_length=None):
        spect = self.spectrogram_function.transform(audio, return_magnitude=True).transpose(2, 3)  # b, c, t, f
        b, c, t, f = spect.shape

        hop_length = hop_length if hop_length is not None else self.patch_length // 2
        hop_ratio = self.patch_length / hop_length
        assert hop_ratio.is_integer()  # the hop ratio should be smaller than patch length and exactly divisible

        left_pad = self.patch_length
        right_pad = 2 * self.patch_length - (t % self.patch_length)
        spect_padded = F.pad(spect, (0, 0, left_pad, right_pad))

        # final_output = self.process_spects(spect_padded)

        fold_params = dict(kernel_size=(self.patch_length, f), dilation=1,
                           padding=0, stride=(hop_length, f))

        chunks = spect_padded.unfold(2, self.patch_length, hop_length).permute(0, 1, 4, 3, 2).contiguous()

        # chunks = torch.nn.Unfold(**fold_params)(spect)  # b, c*patch_length*f, nb_chunks
        nb_chunks = chunks.shape[-1]
        processed_chunks = []
        for chunk_idx in range(nb_chunks):
            chunk = chunks[..., chunk_idx]  # b, c, patch_length, f
            out = self.process_spects(chunk).view(b, -1) / hop_ratio  # process and reshape to b, c*patch_length*f
            processed_chunks.append(out)

        # b, c*patch_length*f, nb_chunks
        processed_chunks = torch.stack(processed_chunks, dim=-1)
        # print(spect.shape, spect_padded.shape, chunks.shape, processed_chunks.shape)
        final_output = torch.nn.Fold(output_size=spect_padded.shape[2:],
                                     **fold_params)(processed_chunks)  # b, c, patch_length, f
        final_output = final_output[:, :, left_pad:-right_pad]
        return final_output
