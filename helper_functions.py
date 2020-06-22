import os


def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)
        
        
def scale_keypoints(keypoints, scale):
    scaled_keypoints = {}
    
    for keypoint_index, keypoint_coords in keypoints.items():
        scaled_keypoints[keypoint_index] = [
            int(keypoint_coords[0] * scale),
            int(keypoint_coords[1] * scale),
        ]
        
    return scaled_keypoints