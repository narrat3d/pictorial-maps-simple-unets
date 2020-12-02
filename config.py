import os

WEIGHTS_FILE_NAME = "weights.hdf5"
CSV_LOG_FILE_NAME = "training.csv"
PLOT_FILE_NAME = "plot.png"

MASK_CHANNEL = 0
NUMBER_OF_KEYPOINTS = 16
NUMBER_OF_BODY_PARTS = 6

KEYPOINT_INDEX = 1 + NUMBER_OF_BODY_PARTS
NUMBER_OF_CHANNELS = 1 + NUMBER_OF_BODY_PARTS + NUMBER_OF_KEYPOINTS + 1


MIRRORED_BODY_PARTS = [0, 1, 5, 4, 3, 2]
MIRRORED_JOINTS = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

MIRRORED_BODY_PARTS = list(map(lambda x: x + 1, MIRRORED_BODY_PARTS))
MIRRORED_JOINTS = list(map(lambda x: x + 1 + len(MIRRORED_BODY_PARTS), MIRRORED_JOINTS))

MIRRORED_CHANNELS = [0] + MIRRORED_BODY_PARTS + MIRRORED_JOINTS + [23]

RUNS = ["1st", "2nd", "3rd"] # "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11st", "12nd"
DATASETS = ["real", "mixed", "synthetic"]

DATASET_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data"
LOG_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs\comparison"
TEST_DATASET = "test"

def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)