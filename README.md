# iSeparate
This repository consists of an attempt to reimplement,
reproduce and unify
the various deep learning based methods for Music
Source Separation.

This project was started as part of the requirement for
the course [Media Computing in Practice](https://media-comp.github.io/2022/) at the University of Tokyo, under the guidance
of [Yusuke Matsui](https://yusukematsui.me/) sensei.

This is a work in progress, current results are decent but not as good as reported in the papers, please use with a pinch of salt.
Will continue to try and improve the quality of separation.

## Currently implemented methods:
|   Model   |                                                                                                                                                Paper                                                                                                                                                |                                       Official code                                        |
|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
|   D3Net   | [Densely connected multidilated convolutional networks for dense prediction tasks <br />(CVPR 2021, Takahashi et al., Sony)](https://openaccess.thecvf.com/content/CVPR2021/papers/Takahashi_Densely_Connected_Multi-Dilated_Convolutional_Networks_for_Dense_Prediction_Tasks_CVPR_2021_paper.pdf) | [link](https://github.com/sony/ai-research-code/tree/master/d3net/music-source-separation) |
| Demucs v2 |                                                                    [Music Source Separation in the Waveform Domain <br />(Arxiv 2021, Defossez et al., Facebook, INRIA)](https://hal.archives-ouvertes.fr/hal-02379796/document)                                                                    |                 [link](https://github.com/facebookresearch/demucs/tree/v2)                 |

## Getting Started
### For Linux users:
Install the [libsndfile](http://www.mega-nerd.com/libsndfile/) and
[soundstretch](https://www.surina.net/soundtouch/soundstretch.html) libraries using your packagemanager, for example:

  ```shell
    sudo apt-get install libsndfile1 soundstretch
  ```
### For Windows and Linux users
If you use anaconda or miniconda, you can quickly create an environment using the provided environment yaml files.

For GPU machines:

```shell
conda env create --name <envname> --file=environment-cuda.yml
```

For CPU only machines:

```shell
conda env create --name <envname> --file=environment-cpu.yml
```

After creating the environment you can activate it as below:

```shell
conda activate <envname>
```

### For Mac users:
  To do

## Separate using pre-trained model
### Create your own Karaoke tracks!
**Currently the D3Net vocals model has been uploaded to Huggingface** and you can
run vocals-accompaniment separation
using that model with the `separate.py` script. Invoke the separation as follows:

```shell
python separate.py \
                -c configs/d3net/eval.yaml \
                -i path/to/song.wav
```
Currently only `.wav` files are supported on windows.
You can use the following command to convert `.mp3` file to `.wav` file within the conda environment created above:

```
ffmpeg -i song.mp3 song.wav
```
You can use `.mp3` files directly on linux, without conversion.
## Dataset Preparation and Training
If you would like to train the models yourself, please follow the following procedure
### Dataset Preparation (MUSDB18)
iSeparate currently supports the [MUSDB18](https://zenodo.org/record/1117372#.Ymcqr9rP1PY) dataset.
This dataset is in the [Native Instruments STEMS](https://www.native-instruments.com/en/specials/stems/) format.
However, it is easier to deal with decded wav files. To do that you can run the `prepare_musdb_dataset.py` file.

If you would like to download a small 7s version of the dataset for testing the code, run

```shell
python prepare_musdb_dataset.py \
                        --root data/MUSDB18-sample \
                        --wav-root data/MUSDB18-sample-wav \
                        --filelists-dir filelists/musdb-sample \
                        --download-sample \
                        --keep-wav-only \
                        --make-symlink
```

If you would like to download the full dataset for training, run

```shell
python prepare_musdb_dataset.py \
                        --root data/MUSDB18 \
                        --wav-root data/MUSDB18-wav \
                        --filelists-dir filelists/musdb \
                        --keep-wav-only \
                        --make-symlink
```

The `prepare_musdb_dataset.py` downloads the data in STEMS format to the directory specified by `--root` and then extracts the
wav files into the directory specified by `--wav-root`. If you want to delete the STEMS and keep only the wav files,
you can use the `--keep-wav-only` option. The `--make-symlink` option will create a symbolic link from the wav directory to the `data/MUSDB18-wav`
directory. If you wanted you could also edit the config files in `configs` directory to point to the dataset directory.
### Dataset Preparation (MTASS)
MTASS is an open-source dataset in which mixtures contain three types of audio signals, speech, music and noise. ([Original Repo](https://github.com/Windstudent/Complex-MTASSNet/))  
iSeparate currently doesn't support MTASS dataset, but the data preparation code is implemented.  
If you modify `training.py` and config files, you can train the model on MTASS dataset.  

If you would like to prepare MTASS dataset, run
```
python prepare_mtass_dataset.py \
                        --root data/MTASS \
                        --wav-root data/MTASS-wav \
                        --filelists-dir filelists/mtass \
                        --keep-wav-only \
                        --make-symlink
```
Options are same as those of musdb datset preparation code.

### Training
Nvidia GPU's are required for training. These models require quite a lot of VRAM, you can change the `batch_size`
parameter in the configs to suit your needs.

Add the `--debug` flag at the end if you just want to do a debug run (train on one batch and validation and then cleans up after itself)

To train on a single GPU:

```shell
python train.py --config-file configs/<method>/<config-name.yaml>
```

To train on multiple GPU with DistributedDataParallel

```shell
python -m torch.distributed.run \
               --nproc_per_node=4 train.py \
               --config-file configs/<method>/<config-name.yaml>
```

## Extending and Contributing
If you would like to add a new method and train on the MUSDB18 dataset, do the following steps

   - create a model package: `models/awesome-method`
        - implement your model
        - add the `separate.py` file and implement the `load_models` and `separate` functions
        - add the model to `model_switcher.py`
   - create and/or add your custom loss functions to the `losses/loss_switcher.py`
   - create config files following the examples in `configs` directory
