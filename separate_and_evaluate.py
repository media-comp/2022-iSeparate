import argparse
import os
import shutil
import sys
import functools
import multiprocessing

import musdb
import museval
import torch
import tqdm
import yaml

from models.model_switcher import get_separation_funcs
from utils.common_utils import preprocess_audio


def separate_and_evaluate(
        _track,
        _model_name,
        _model_path,
        _output_dir,
        _eval_dir,
        _mus,
        eval_params,
):
    model_loader, separator = get_separation_funcs(_model_name)
    model = model_loader(_model_path)
    audio = torch.as_tensor(_track.audio, dtype=torch.float32)
    audio = preprocess_audio(audio, _track.rate, model)

    estimates = separator(
        model,
        audio,
        **eval_params
    )

    for key in estimates:
        estimates[key] = estimates[key].cpu().detach().numpy().T
    if _output_dir:
        _mus.save_estimates(estimates, _track, _output_dir)

    _scores = museval.eval_mus_track(_track, estimates, output_dir=_eval_dir)
    return _scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="path to the evaluation config file")
    parser.add_argument('--debug', action='store_true',
                        help="do a debug run")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg['output_dir'] = 'debug_results/estimates'
        cfg['eval_dir'] = 'debug_results/scores'
        cfg['nproc'] = 1

    mus = musdb.DB(**cfg['musdb_config'])

    if cfg['nproc'] > 1:
        pool = multiprocessing.Pool(cfg['nproc'])
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    _model_name=cfg['model_name'],
                    _model_path=cfg['model_path'],
                    _output_dir=cfg['output_dir'],
                    _eval_dir=cfg['eval_dir'],
                    _mus=mus,
                    eval_params=cfg['eval_params'],
                ),
                iterable=mus.tracks,
                chunksize=1,
            )
        )
        pool.close()
        pool.join()
        for scores in scores_list:
            results.add_track(scores)

    else:
        results = museval.EvalStore()
        for track in tqdm.tqdm(mus.tracks):
            scores = separate_and_evaluate(
                track,
                _model_name=cfg['model_name'],
                _model_path=cfg['model_path'],
                _output_dir=cfg['output_dir'],
                _eval_dir=cfg['eval_dir'],
                _mus=mus,
                eval_params=cfg['eval_params'],
            )
            print(track, "\n", scores)
            results.add_track(scores)
            if args.debug:
                break

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, cfg['model_name'])
    method.save(os.path.join(cfg['eval_dir'], cfg['model_name'] + ".pandas"))

    if args.debug:
        shutil.rmtree('debug_results')
