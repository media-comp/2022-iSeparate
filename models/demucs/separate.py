import torch

from models.demucs.demucs import Demucs


def load_models(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    model = Demucs(**config.model_args)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.eval()
    model = {
        'all': {
            'model': model,
            'config': config
        }
    }
    print('Loaded model from iteration: {}'.format(checkpoint['iteration']))
    return model


@torch.no_grad()
def separate(
        model,
        audio,
        cpu_only,
        shifts
):
    device = 'cuda' if torch.cuda.is_available() and not cpu_only else 'cpu'

    mix_audio = audio.to(device)
    if len(mix_audio.shape) == 2:
        mix_audio = mix_audio.unsqueeze(0)  # add batch dimension if only single audio tensor is provided
    config = model['all']['config']
    model = model['all']['model'].to(device)
    output = model.separate(mix_audio, shifts, sample_rate=config.data_loader_args['train']['sample_rate'])

    targets = config.data_loader_args['train']['target_sources']
    estimates = {}
    for i, target in enumerate(targets):
        estimates[target] = output[0, i]

    estimates['accompaniment'] = mix_audio.squeeze(0) - estimates['vocals']

    return estimates
