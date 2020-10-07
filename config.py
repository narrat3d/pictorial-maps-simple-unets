import os


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


def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)