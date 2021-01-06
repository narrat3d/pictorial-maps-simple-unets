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
ARCHITECTURES = ["simple_deconv", "simple_unet", "simple_unet+", "simple_unet++"]

DATASETS = ["real", "mixed", "synthetic"]

DATASET_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data"
LOG_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs"
TEST_DATASET = "test"

COLORS = [
    (255, 255, 255), # background
    (255, 242, 0), # torso
    (34, 177, 76), # head
    (136, 0, 21), # right arm
    (63, 72, 204), # right leg
    (0, 162, 232), # left leg
    (237, 28, 36), # left arm
]

# KEYPOINTS
ankle_right = 0
knee_right = 1
hip_right = 2
hip_left = 3
knee_left = 4
ankle_left = 5
hip = 6
thorax = 7
neck = 8
head = 9
wrist_right = 10
elbow_right = 11
shoulder_right = 12
shoulder_left = 13
elbow_left = 14
wrist_left = 15

BONES = [
    [ankle_right, knee_right, COLORS[4]], 
    [knee_right, hip_right, COLORS[4]], 
    # [hip_right, hip],
    [ankle_left, knee_left, COLORS[5]], 
    [knee_left, hip_left, COLORS[5]], 
    # [hip_left, hip],
    [hip, thorax, COLORS[1]], 
    [thorax, neck, COLORS[1]], 
    [neck, head, COLORS[2]],
    [wrist_right, elbow_right, COLORS[3]], 
    [elbow_right, shoulder_right, COLORS[3]], 
    # [shoulder_right, neck],
    [wrist_left, elbow_left, COLORS[6]], 
    [elbow_left, shoulder_left, COLORS[6]], 
    # [shoulder_left, neck], 
];

def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)