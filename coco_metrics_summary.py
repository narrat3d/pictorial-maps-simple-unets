'''
input:
20 runs for different architectures

output:
calculate the average of scores from these runs
see which run achieved the highest COCO scores

purpose:
needed for the research paper
'''
import os
import json
import numpy as np

from config import METRICS_LOG_FILE_NAME, get_architecture_folder, ARCHITECTURES


def print_result(coco_array):
    results_string = "\t".join(map(lambda result: str(round(result, 2)) + "%", coco_array))
    print (results_string)


def aggregate_results(log_folder):
    result_folders = os.listdir(log_folder)
    
    aggregated_results = {}

    for result_folder in result_folders:
        dataset_name = result_folder.split("_")[0]
        results_for_dataset = aggregated_results.setdefault(dataset_name, {})
        bodypart_results_for_dataset = results_for_dataset.setdefault("bodyparts", [])
        keypoint_results_for_dataset = results_for_dataset.setdefault("keypoints", [])  
        
        metrics_log_file_path = os.path.join(log_folder, result_folder, METRICS_LOG_FILE_NAME)
        
        with open(metrics_log_file_path) as metrics_log_file:
            logged_metrics = json.load(metrics_log_file)
        
        bodypart_results_for_dataset.append(logged_metrics["coco"]["bodyparts"])
        keypoint_results_for_dataset.append(logged_metrics["coco"]["keypoints"])
    
    for dataset in aggregated_results:
        print (dataset)
        
        for task in ["bodyparts", "keypoints"]:
            results_array = aggregated_results[dataset][task]
                        
            results_array.sort(key=lambda array: array[0])
            filtered_results_array = results_array[4:-4]
            
            average_results = np.mean(np.array(filtered_results_array), axis=0).tolist()
            print_result(average_results)


def find_best_result(log_folder):
    result_folders = os.listdir(log_folder)
    
    best_ap = {
        "bodyparts": [0] * 6,
        "keypoints": [0] * 6,
        "folder": None
    }
    
    lowest_error = {
        "bodyparts": 100000,
        "keypoints": 100000,
        "folder": None
    }

    for result_folder in result_folders:
        metrics_log_file_path = os.path.join(log_folder, result_folder, METRICS_LOG_FILE_NAME)
        
        with open(metrics_log_file_path) as metrics_log_file:
            logged_metrics = json.load(metrics_log_file)
        
        keypoints_ap = logged_metrics["coco"]["keypoints"]
        bodyparts_ap = logged_metrics["coco"]["bodyparts"]
        keypoints_error = logged_metrics["error"]["keypoints"]
        bodyparts_error = logged_metrics["error"]["bodyparts"]        
        
        
        if (keypoints_ap[0] + bodyparts_ap[0] > 
            best_ap["keypoints"][0] + best_ap["bodyparts"][0]):
            best_ap["keypoints"] = keypoints_ap
            best_ap["bodyparts"] = bodyparts_ap
            best_ap["folder"] = os.path.basename(result_folder)
            
        if (keypoints_error + bodyparts_error < 
            lowest_error["keypoints"] + lowest_error["bodyparts"]):
            lowest_error["keypoints"] = keypoints_error
            lowest_error["bodyparts"] = bodyparts_error
            lowest_error["folder"] = os.path.basename(result_folder)
    
    print ("highest precision: %s" % best_ap["folder"])
    print_result(best_ap["bodyparts"])
    print_result(best_ap["keypoints"])
    
    print ("lowest error: %s" % lowest_error["folder"])
    print ("%s%%" % lowest_error["bodyparts"])
    print ("%s%%" % lowest_error["keypoints"])
    
    
if __name__ == '__main__':
    for architecture in ARCHITECTURES: 
        print(architecture + "\n") 
        architecture_folder = get_architecture_folder(architecture)
        
        aggregate_results(architecture_folder)
        find_best_result(architecture_folder)
        print("\n")