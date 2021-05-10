# Body part parsing and pose estimation of single human figures with simplified UNet++ versions

## Installation

* Requires [Python 3.7.x](https://www.python.org/downloads/) or higher
* Requires [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
* Download [this project](https://gitlab.ethz.ch/narrat3d/pictorial-maps-simple-unets/-/archive/master/pictorial-maps-mask-rcnn-master.zip)
* pip install -r requirements.txt

## Training
* Download the [training data](https://ikgftp.ethz.ch/?u=K8bH&p=3RwE&path=/human_figures_training_data.zip) and [test data](https://ikgftp.ethz.ch/?u=VDYk&p=Bm6D&path=/human_figures_test_data.zip), store them in the same folder and set DATASET_FOLDER to its location in config.py
* Set LOG_FOLDER in config.py where intermediate snapshots shall be stored
* Optionally adjust properties like datasets (e.g. separated), architectures (e.g. simple-unet+), and number of runs (e.g. 1st)
* Run train_and_eval_wrapper.py
* If you enable the DEBUG variable in config.py, the CNN is trained only on the first 10 images, which are also used for validation. By this, it can be tested if a network is able to learn. 

## Evaluation
* Once a training run finishes, the best model of all epochs will be automatically evaluated
* Optionally calculate custom error metrics of a single run in error_metrics.py
* Optionally calculate the average precision of a single run in coco_metrics.py 
* Scores of multiple runs can be aggregated and the highest score can be determined in coco_metrics_summary.py

## Inference
* Download the [pre-trained model](https://ikgftp.ethz.ch/?u=cndC&p=tWwd&path=/human_figures_model.zip) and set the INFERENCE_MODEL_FOLDER in config.py to its location
* Run inference.py

## Tested architectures
![Implemented architectures](architectures.png "Implemented architectures")

© 2020-2021 ETH Zurich, Raimund Schnürer