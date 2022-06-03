import yaml
from torch.utils.data import DataLoader

from iSeparate.datasets.sep_dataset import SeparationDataset


def test_dataset_demucs():
    with open("configs/demucs/demucs_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # cfg['data_loader_args']['train']['data_root'] = os.path.join('.', cfg['data_loader_args']['train']['data_root'])
    # cfg['data_loader_args']['validation']['data_root'] = os.path.join('.', cfg['data_loader_args']['validation']['data_root'])
    # cfg['data_loader_args']['train']['song_lists'] = [os.path.join('.', f) for f in cfg['data_loader_args']['train']['song_lists']]
    # cfg['data_loader_args']['validation']['song_lists'] = [os.path.join('.', f) for f in cfg['data_loader_args']['validation']['song_lists']]

    cfg["data_loader_args"]["train"]["data_root"] = "data/MUSDB18-sample-wav"
    cfg["data_loader_args"]["validation"]["data_root"] = "data/MUSDB18-sample-wav"
    cfg["data_loader_args"]["train"]["song_lists"] = ["filelists/musdb-sample/train.txt"]
    cfg["data_loader_args"]["validation"]["song_lists"] = ["filelists/musdb-sample/validation.txt"]

    d_train = SeparationDataset(**cfg["data_loader_args"]["train"])
    d_val = SeparationDataset(**cfg["data_loader_args"]["validation"])

    train_loader = DataLoader(d_train, batch_size=1, num_workers=1, drop_last=True)
    val_loader = DataLoader(d_val, batch_size=1, num_workers=1, drop_last=True)

    for _ in train_loader:
        break

    for _ in val_loader:
        break


def test_dataset_d3net():
    with open("configs/d3net/vocals.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["data_loader_args"]["train"]["data_root"] = "data/MUSDB18-sample-wav"
    cfg["data_loader_args"]["validation"]["data_root"] = "data/MUSDB18-sample-wav"
    cfg["data_loader_args"]["train"]["song_lists"] = ["filelists/musdb-sample/train.txt"]
    cfg["data_loader_args"]["validation"]["song_lists"] = ["filelists/musdb-sample/validation.txt"]

    d_train = SeparationDataset(**cfg["data_loader_args"]["train"])
    d_val = SeparationDataset(**cfg["data_loader_args"]["validation"])

    train_loader = DataLoader(d_train, batch_size=1, num_workers=15, drop_last=True)
    val_loader = DataLoader(d_val, batch_size=1, num_workers=15, drop_last=True)

    for _ in train_loader:
        break

    for _ in val_loader:
        break
