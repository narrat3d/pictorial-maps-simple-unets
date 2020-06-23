'''
precondition:
predicted masks and keypoints with training.py

input:
folder with groundtruth images, masks and keypoints
folder with detected masks and keypoints

output:
average normalized false positives and false negatives for each body part
average normalized errors for each joint

purpose:
alternative metrics to COCO
'''

import math
import json
from PIL import Image
from training import NUMBER_OF_KEYPOINTS, NUMBER_OF_BODY_PARTS, MASK_CHANNEL
import numpy as np
import os

BODY_PART_NAMES = [
    "torso",
    "head",
    "right arm",
    "right leg",
    "left leg",
    "left arm"
]

KEYPOINT_NAMES = [
    "right ankle",
    "right knee",
    "right hip",
    "left hip",
    "left knee",
    "left ankle",
    "hip",
    "thorax",
    "neck",
    "head",
    "right wrist",
    "right elbow",
    "right shoulder",
    "left shoulder",
    "left elbow",
    "left wrist"
]

def avg(errors):
    error_sum = 0
    
    for error in errors:
        error_sum += error
        
    return error_sum / len(errors)


def add_keypoints_error(gt_keypoints, detected_keypoints, image_size, keypoint_errors):
    for keypoint_index in range(NUMBER_OF_KEYPOINTS):
        gt_keypoint = gt_keypoints.get(str(keypoint_index))
        detected_keypoint = detected_keypoints.get(str(keypoint_index))
        
        if (gt_keypoint == None and detected_keypoint == None):
            error = 0
        elif ((gt_keypoint == None and detected_keypoint != None) or 
            (gt_keypoint != None and detected_keypoint == None)):
            error = 1
        else :
            # normalize
            x = (gt_keypoint[0] - detected_keypoint[0]) / image_size
            y = (gt_keypoint[1] - detected_keypoint[1]) / image_size
            error = math.hypot(x, y)
        
        keypoint_errors[keypoint_index].append(error)


def add_masks_error(gt_masks, predicted_masks, image_size, masks_errors):
    image_area = image_size * image_size
    
    for mask_index in range(NUMBER_OF_BODY_PARTS):
        gt_mask = gt_masks[mask_index]
        predicted_mask = predicted_masks[mask_index]
        
        false_positives = ((predicted_mask - gt_mask) == 1).astype(int)
        false_negatives = ((gt_mask - predicted_mask) == 1).astype(int)
        
        false_positives_number = np.sum(false_positives)
        false_negatives_number = np.sum(false_negatives)
        
        false_positives_normalized = false_positives_number / image_area
        false_negatives_normalized = false_negatives_number / image_area
        
        mask_error = false_positives_normalized + false_negatives_normalized
        masks_errors[mask_index].append(mask_error)


def hot_encode_mask(mask):
    mask_channel = mask.getchannel(MASK_CHANNEL)
    masks = []
    
    for mask_index in range(NUMBER_OF_BODY_PARTS):
        mask_np = np.asarray(mask_channel)
        mask_np = (mask_np == mask_index).astype(int)
        masks.append(mask_np)
        
    return masks


def evaluate_single_image(ground_truth_folder, results_folder, image_name, keypoint_errors, body_part_errors):
    image_name_without_ext, _ = os.path.splitext(image_name)
    
    keypoints_gt_path = os.path.join(ground_truth_folder, "keypoints", image_name_without_ext + ".json")
    keypoints_results_path = os.path.join(results_folder, "keypoints", image_name_without_ext + ".json")
    
    mask_gt_path = os.path.join(ground_truth_folder, "masks", image_name_without_ext + ".png")
    mask_results_path = os.path.join(results_folder, "masks", image_name_without_ext + ".png")
    
    if (not os.path.exists(keypoints_results_path)):
        return
    
    mask_gt_image = Image.open(mask_gt_path)
    predicted_body_parts = Image.open(mask_results_path)
    
    image_size = mask_gt_image.height # equals also width
    
    gt_masks = hot_encode_mask(mask_gt_image)
    predicted_masks = hot_encode_mask(predicted_body_parts)
    
    add_masks_error(gt_masks, predicted_masks, image_size, body_part_errors)
    
    gt_keypoints = json.load(open(keypoints_gt_path))
    detected_keypoints = json.load(open(keypoints_results_path))
    
    add_keypoints_error(gt_keypoints, detected_keypoints, image_size, keypoint_errors)

    
def evaluate(ground_truth_folder, results_folder): 
    keypoint_errors = {}
    body_part_errors = {}
    
    for i in range(NUMBER_OF_KEYPOINTS):
        keypoint_errors[i] = []
        
    for i in range(NUMBER_OF_BODY_PARTS):
        body_part_errors[i] = []
    
    images_folder = os.path.join(ground_truth_folder, "images")

    for image_name in os.listdir(images_folder):
        evaluate_single_image(ground_truth_folder, results_folder, image_name, keypoint_errors, body_part_errors)    
    
    print("Body part errors:")
    
    for i in range(NUMBER_OF_BODY_PARTS):
        body_part_name = BODY_PART_NAMES[i]
        rounded_error = round(avg(body_part_errors[i]) * 100, 2)
        print (body_part_name, rounded_error)
    
    print()
    print("Keypoint errors:")
    
    for i in range(NUMBER_OF_KEYPOINTS):
        keypoint_name = KEYPOINT_NAMES[i]
        rounded_error = round(avg(keypoint_errors[i]) * 100, 2)
        print (keypoint_name, rounded_error)
        
if __name__ == '__main__':
    evaluate(r"E:\CNN\masks\data\figures\real", r"E:\CNN\logs\body_parts\real")