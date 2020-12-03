import os

WEIGHTS_FILE_NAME = "weights.hdf5"
CSV_LOG_FILE_NAME = "training.csv"
PLOT_LOG_FILE_NAME = "plot.png"
METRICS_LOG_FILE_NAME = "metrics.json"

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

DEBUG = False
MIRROR_IMAGES = False

# input height and width for images into the network
IMAGE_SIZE = 128
# e.g. factor of 2 leads to predicted masks of size 128px
MASK_DOWNSAMPLING_FACTOR = 2
# standard deviation of gaussian kernel for creating keypoint heatmaps
SIGMA = 2

RUNS = ["1st", "2nd", "3rd", "4th", "5th"]
DATASETS = ["real", "mixed", "synthetic"]
ARCHITECTURES = ["simple_unet+"] # "simple_deconv", "simple_unet", "simple_unet++"

# RUNS = ["1st"]
DATASETS = ["real", "mixed", "synthetic"]

DATASET_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data"
LOG_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs"
TEST_DATASET = "test"

COLORS = [
    (255, 255, 255),
    (255, 215, 0), # torso
    (255, 119, 255), # head
    (191, 255, 0), # right arm
    (255, 0, 0), # right hand
    (176, 48, 96), # right leg
    (0, 0, 128), # right foot
    (255, 228, 196), # left leg
    (0, 191, 255), # left foot
    (0, 100, 0), # left arm
    (0, 0, 0) # left hand
]

def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)