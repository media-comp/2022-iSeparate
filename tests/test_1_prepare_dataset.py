import subprocess


def test_prepare_dataset():
    subprocess.call(
        [
            "python",
            "prepare_dataset.py",
            "--root",
            "data/MUSDB18-sample",
            "--wav-root",
            "data/MUSDB18-sample-wav",
            "--filelists-dir",
            "filelists/musdb-sample",
            "--download-sample",
            "--keep-wav-only",
        ]
    )
