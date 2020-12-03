'''
precondition:
predicted masks and keypoints with training.py

input: 
COCO files with groundtruth masks and keypoints
folder with detected masks and keypoints

output:
average precision for joints 
average precision for body parts 

purpose:
calculate COCO metrics 
'''

import numpy as np
from PIL import Image
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
# import skimage.io as io
# import matplotlib.pyplot as plt
from config import NUMBER_OF_BODY_PARTS, MASK_CHANNEL, METRICS_LOG_FILE_NAME

# source: https://github.com/facebookresearch/Detectron/issues/640
KEYPOINT_MAPPING = {
    "17": 0, # right_ankle
    "15": 1, # right_knee
    "13": 2, # right_hip
    "12": 3, # left_hip
    "14": 4, # left_knee
    "16": 5, # left_ankle
    "1": 9, # nose
    "11": 10, # right_wrist
    "9": 11, # right_elbow
    "7": 12, # right_shoulder
    "6": 13, # left_shoulder
    "8": 14, # left_elbow,
    "10": 15, # left_wrist
    # "2": 0, # left_eye
    # "3": 0, # right_eye
    # "4": 0, # left_ear
    # "5": 0 # right_ear
}

COCO_KEYPOINT_NUMBER = 17

image_ids = {}


def add_ground_truth(coco_ground_truth_path):
    coco = COCO(coco_ground_truth_path)
    
    annIds = coco.getAnnIds(imgIds=1)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)


# for debugging one single image
"""
def show_ground_truth(input_image_path, output_image_path, keypoints_ground_truth_path, masks_ground_truth_path):
    I = io.imread(input_image_path)
    plt.axis('off')
    plt.imshow(I)
    
    add_ground_truth(keypoints_ground_truth_path)
    add_ground_truth(masks_ground_truth_path)
    
    plt.savefig(output_image_path)
"""

def load_coco_data(file_path):
    coco_data = json.load(open(file_path))
    
    coco_images = coco_data["images"]
    
    for coco_image in coco_images:
        image_ids[coco_image["file_name"]] = coco_image["id"]


def coco_keypoint_result(image_id, detected_keypoints):
    coco_keypoints = []
        
    for i in range(COCO_KEYPOINT_NUMBER):
        index = KEYPOINT_MAPPING.get(str(i + 1), -1)
        coords = detected_keypoints.get(str(index))
        # coords = detected_keypoints.get(i)
        
        if (coords == None):
            x = 0
            y = 0
            visibility = 0
        else :
            x = coords[0]
            y = coords[1]
            visibility = 1
        
        coco_keypoints.append(x)
        coco_keypoints.append(y)
        coco_keypoints.append(visibility)
        
    return {
        "image_id": image_id, 
        "category_id": 0, 
        "keypoints": coco_keypoints, 
        "score": 1.0,
    }


def coco_mask_result(image_id, detected_mask):
    coco_masks = []
    
    for i in range(NUMBER_OF_BODY_PARTS):
        mask = detected_mask.copy()
        binary_mask = mask.point(lambda p: (p == i and 1) or 0)
        
        mask_np = np.asfortranarray(binary_mask)
        
        mask_rle = COCOmask.encode(mask_np)
        mask_rle["counts"] = mask_rle["counts"].decode('utf-8')
        
        coco_masks.append({
            "image_id": image_id, 
            "category_id": i + 1, 
            "segmentation": mask_rle, 
            "area": COCOmask.area(mask_rle).tolist(), 
            "bbox": COCOmask.toBbox(mask_rle).tolist(), 
            "score": 1.0,        
        })
         
    return coco_masks


def calculate_results(coco_ground_truth_path, coco_results_path, ann_type):
    cocoGt=COCO(coco_ground_truth_path)
    cocoDt=cocoGt.loadRes(coco_results_path)
    
    coco_data = json.load(open(coco_results_path))
    imgIds = list(set(map(lambda a: a["image_id"], coco_data)))
    
    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return (cocoEval.stats * 100).tolist()[0:6]


def create_keypoint_result_file(keypoints_ground_truth_path, output_folder):
    coco_keypoints = []
    
    load_coco_data(keypoints_ground_truth_path)
    
    for json_file_name in os.listdir(output_folder):
        image_file_name = json_file_name.replace(".json", ".jpg")
        image_id = image_ids.get(image_file_name)
    
        if (image_id == None):
            continue
        
        json_file_path = os.path.join(output_folder, json_file_name)
        keypoints = json.load(open(json_file_path))
        
        coco_keypoint_result_data = coco_keypoint_result(image_id, keypoints)
        coco_keypoints.append(coco_keypoint_result_data)
    
    results_keypoint_file_path = os.path.join(output_folder, "coco_keypoints.json")
    
    json_file = open(results_keypoint_file_path, "w")
    json.dump(coco_keypoints, json_file)
    
    return results_keypoint_file_path


def create_mask_result_file(masks_ground_truth_path, output_folder):
    coco_masks = []
    
    load_coco_data(masks_ground_truth_path)
    
    for mask_file_name in os.listdir(output_folder):
        image_file_name = mask_file_name.replace(".png", ".jpg")
        image_id = image_ids.get(image_file_name)
    
        if (image_id == None):
            continue
        
        mask_file_path = os.path.join(output_folder, mask_file_name)
        mask = Image.open(mask_file_path).getchannel(MASK_CHANNEL)
        
        coco_mask_result_data = coco_mask_result(image_id, mask)
        coco_masks.extend(coco_mask_result_data)
        
    results_mask_file_path = os.path.join(output_folder, "coco_masks.json")
    
    json_file = open(results_mask_file_path, "w")
    json.dump(coco_masks, json_file)
    
    return results_mask_file_path


def aggregate_results(log_folder):
    result_folders = os.listdir(log_folder)
    
    aggregated_results = {}

    for result_folder in result_folders:
        dataset_name = result_folder.split("_")[0]
        results_for_dataset = aggregated_results.setdefault(dataset_name, {})
        keypoint_results_for_dataset = results_for_dataset.setdefault("keypoints", [])
        bodypart_results_for_dataset = results_for_dataset.setdefault("bodyparts", [])
        
        metrics_log_file_path = os.path.join(log_folder, result_folder, METRICS_LOG_FILE_NAME)
        
        with open(metrics_log_file_path) as metrics_log_file:
            logged_metrics = json.load(metrics_log_file)
            
        keypoint_results_for_dataset.append(logged_metrics["coco"]["keypoints"])
        bodypart_results_for_dataset.append(logged_metrics["coco"]["bodyparts"])
    
    for dataset in aggregated_results:
        print (dataset)
        
        for task in ["keypoints", "bodyparts"]:
            results_array = aggregated_results[dataset][task]
                        
            results_array.sort(key=lambda array: array[0])
            filtered_results_array = results_array[1:-1]
            
            average_results = np.mean(np.array(filtered_results_array), axis=0).tolist()
            
            average_results_string = "\t".join(map(lambda result: str(round(result, 2)) + "%", average_results))
            print (average_results_string)


def main(ground_truth_folder, results_folder):
    keypoints_ground_truth_path = os.path.join(ground_truth_folder, "coco_keypoints.json")
    masks_ground_truth_path = os.path.join(ground_truth_folder, "coco_masks.json")
    
    keypoints_results_folder = os.path.join(results_folder, "keypoints")
    masks_results_folder = os.path.join(results_folder, "masks")
    
    keypoints_results_path = create_keypoint_result_file(keypoints_ground_truth_path, keypoints_results_folder)
    keypoints_avarage_precisions = calculate_results(keypoints_ground_truth_path, keypoints_results_path, "keypoints")
    
    masks_results_path = create_mask_result_file(masks_ground_truth_path, masks_results_folder)
    masks_avarage_precisions = calculate_results(masks_ground_truth_path, masks_results_path, "segm")
    
    return [keypoints_avarage_precisions, masks_avarage_precisions]


if __name__ == '__main__':
    r"""
    ground_truth_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data\test"
    results_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs\unet++\mixed_1st"
    
    metrics = main(ground_truth_folder, results_folder)
    print(metrics)
    """
    log_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs\unet++"
    aggregate_results(log_folder)