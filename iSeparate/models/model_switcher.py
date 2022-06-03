def get_model(model_name, model_args, input_mean, input_scale):
    if model_name == "D3Net":
        from iSeparate.models.d3net.d3net import MMD3Net

        model = MMD3Net(**model_args, input_mean=input_mean, input_scale=input_scale)
    elif model_name == "Demucs":
        from iSeparate.models.demucs.demucs import Demucs

        model = Demucs(**model_args)
    else:
        raise Exception("Unknown model!!")

    return model


def get_separation_funcs(model_name):
    model_loader = None
    separator = None
    if model_name == "D3Net":
        from iSeparate.models.d3net.separate import load_models, separate

        model_loader = load_models
        separator = separate
    elif model_name == "Demucs":
        from iSeparate.models.demucs.separate import load_models, separate

        model_loader = load_models
        separator = separate
    return model_loader, separator
