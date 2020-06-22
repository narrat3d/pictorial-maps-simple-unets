'''
Created on 19.06.2020

@author: raimund
'''
import math
from helper_functions import scale_keypoints
import json
from PIL import Image
from training import NUMBER_OF_KEYPOINTS, NUMBER_OF_BODY_PARTS, IMAGE_SIZE,\
    MASK_DOWNSAMPLING_FACTOR, MASK_CHANNEL, predict
import numpy as np
import os

def avg(errors):
    error_sum = 0
    
    for error in errors:
        error_sum += error
        
    return error_sum / len(errors)

def add_keypoints_error(gt_keypoints, detected_keypoints, image_size, keypoint_errors):
    for keypoint_index in range(NUMBER_OF_KEYPOINTS):
        gt_keypoint = gt_keypoints.get(keypoint_index)
        detected_keypoint = detected_keypoints.get(keypoint_index)
        
        if (gt_keypoint == None and detected_keypoint != None):
            error = 1
        elif (gt_keypoint != None and detected_keypoint == None):
            error = 1
        else :
            x = (gt_keypoint[0] - detected_keypoint[0]) / image_size
            y = (gt_keypoint[1] - detected_keypoint[1]) / image_size
            error = math.hypot(x, y)
        
        keypoint_errors[keypoint_index].append(error)


def add_masks_error(gt_masks, predicted_masks, image_size, masks_errors):
    image_area = image_size * image_size
    
    for mask_index in range(NUMBER_OF_BODY_PARTS + 1):
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


def evaluate_single_image(model, image_path, keypoint_errors, body_part_errors):
    image = Image.open(image_path)
    
    keypoints_path = image_path.replace("images", "keypoints").replace(".jpg", ".json")
    gt_keypoints = json.load(open(keypoints_path))
    gt_keypoints_scaled = scale_keypoints(gt_keypoints, IMAGE_SIZE / image.width)

    mask_size = int(IMAGE_SIZE / MASK_DOWNSAMPLING_FACTOR)
    mask_image = Image.open(image_path.replace("images", "masks").replace(".jpg", ".png"))
    mask_image = mask_image.getchannel(MASK_CHANNEL)
    resized_mask_image = mask_image.resize((mask_size, mask_size), Image.NEAREST)
 
    background_mask_np = np.asarray(resized_mask_image)
    background_mask_np = (background_mask_np == 255).astype(int)
    gt_masks = [background_mask_np]
    
    for mask_index in range(NUMBER_OF_BODY_PARTS):
        mask_np = np.asarray(mask_image)
        mask_np = (mask_np == mask_index).astype(int)
        gt_masks.append(mask_np)
    
    detected_keypoints, predicted_body_parts, _ = predict(model, image)
    
    add_keypoints_error(gt_keypoints_scaled, detected_keypoints, IMAGE_SIZE, keypoint_errors)
    
    predicted_masks = []
    
    for i in range(NUMBER_OF_BODY_PARTS + 1):
        body_part_mask = (predicted_body_parts == i).astype(int)
        predicted_masks.append(body_part_mask)
        
    add_masks_error(gt_masks, predicted_masks, IMAGE_SIZE, body_part_errors)   
    
def evaluate(model, val_dir): 
    keypoint_errors = {}
    body_part_errors = {}
    
    for i in range(NUMBER_OF_KEYPOINTS):
        keypoint_errors[i] = []
        
    for i in range(NUMBER_OF_BODY_PARTS + 1):
        body_part_errors[i] = []
        
    image_names = os.listdir(val_dir)
        
    for image_name in image_names:
        evaluate_single_image(model, os.path.join(val_dir, image_name), keypoint_errors, body_part_errors)    
    
    print ("Body part errors:")
    
    for i in range(NUMBER_OF_BODY_PARTS + 1):
        print (i, avg(body_part_errors[i]))
    
    print ("Keypoint errors:")
    
    for i in range(NUMBER_OF_KEYPOINTS):
        print (i, avg(keypoint_errors[i]))   