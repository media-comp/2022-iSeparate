import norbert
import numpy as np
import torch
import torchaudio.transforms

from models.d3net.d3net import MMD3Net


def load_models(target_model_paths):
    target_models = {}
    for target, model_path in target_model_paths.items():
        # skip targets without model path
        if model_path is None:
            continue
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        model = MMD3Net(**config.model_args)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model = model.eval()
        target_models[target] = {'model': model, 'config': config}
        print('Loaded {} model from iteration: {}'.format(target, checkpoint['iteration']))

    if len(target_models.keys()) == 0:
        raise Exception('Specify atleast one target model path')
    return target_models


@torch.no_grad()
def separate(
        target_models,
        audio,
        niter,
        cpu_only,
        softmask
):
    if len(target_models) == 1:
        assert 'vocals' in target_models, 'For calculating scores, at least vocals target is necessary!'
    device = 'cuda' if torch.cuda.is_available() and not cpu_only else 'cpu'

    mix_audio = audio.to(device)
    if len(mix_audio.shape) == 2:
        mix_audio = mix_audio.unsqueeze(0)  # add batch dimension if only single audio tensor is provided

    mix_stft = None

    nb_sources = len(target_models.keys())

    stft_fn = None
    istft_fn = None
    estimates = {}
    estimates_tensor = None

    for i, (target, model_dict) in enumerate(target_models.items()):
        model = model_dict['model']
        config = model_dict['config']
        model = model.to(device)
        if stft_fn is None:
            stft_fn = torchaudio.transforms.Spectrogram(
                n_fft=config.model_args['n_fft'],
                win_length=config.model_args['win_length'],
                hop_length=config.model_args['hop_length'],
                power=None,
                center=True
            ).to(device)
            istft_fn = torchaudio.transforms.InverseSpectrogram(
                n_fft=config.model_args['n_fft'],
                win_length=config.model_args['win_length'],
                hop_length=config.model_args['hop_length'],
                center=True
            )

            mix_stft = stft_fn(mix_audio).permute(0, 3, 2, 1).squeeze(0)     # bcft --> btfc

        estimate = model.separate(mix_audio, hop_length=128)
        estimate = estimate.permute(0, 2, 3, 1).squeeze(0)  # b,c,t,f --> t,f,c
        if estimates_tensor is None:
            estimates_tensor = estimate.unsqueeze(-1).cpu()
        else:
            estimates_tensor = torch.cat([estimates_tensor, estimate.unsqueeze(-1).cpu()], dim=-1)

        estimates[target] = estimate.cpu().numpy()

    if nb_sources < 4:
        nb_sources += 1
        if 'vocals' in target_models:
            # add the accompaniment key to the estimates directory in case of voc/acc separation
            # add the accompaniment as mix - vocals
            estimates['accompaniment'] = None
            # print(estimates['vocals'].shape)
            estimate_accompaniment = norbert.contrib.residual_model(estimates['vocals'][..., np.newaxis],
                                                                    mix_stft.cpu().numpy())
            estimates['accompaniment'] = estimate_accompaniment[..., 1]
            estimates_tensor = torch.cat([estimates_tensor, torch.from_numpy(estimate_accompaniment[..., 1:])], dim=-1)

    Y = norbert.wiener(estimates_tensor.numpy(), mix_stft.cpu().numpy(), use_softmask=softmask, iterations=niter)

    for i, target in enumerate(estimates.keys()):
        estimates[target] = istft_fn(torch.from_numpy(Y[..., i]).permute(2, 1, 0))     # t f c --> c f t

    return estimates
