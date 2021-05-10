'''
purpose:
train and evaluate different architectures and datasets multiple times
avoid memory leak
'''
from config import ARCHITECTURES, DATASETS, RUNS
import subprocess

for architecture in ARCHITECTURES:
    for dataset in DATASETS:
        for run in RUNS:
            subprocess.run(["python", "train_and_eval.py", architecture, dataset, run])