'''
input:
model weights
name of used architecture

output:
detected body part masks and pose keypoints
evaluation metrics

purpose:
evaluate a pre-trained model
'''
import os
from train_and_eval import create_model,\
    predict_single_image, calculate_metrics
from config import INFERENCE_MODEL_WEIGHTS, BEST_ARCHITECTURE,\
    GROUND_TRUTH_FOLDER, LOG_FOLDER, mkdir_if_not_exists


def main(weights_file_path, architecture, name):
    test_image_folder = os.path.join(GROUND_TRUTH_FOLDER, "images")
    image_names = os.listdir(test_image_folder)

    log_folder = os.path.join(LOG_FOLDER, name)

    keypoint_output_folder = os.path.join(log_folder, "keypoints")
    mask_output_folder = os.path.join(log_folder, "masks")
    
    list(map(mkdir_if_not_exists, [log_folder, keypoint_output_folder, mask_output_folder]))

    model = create_model(architecture)
    model.load_weights(weights_file_path)

    for image_name in image_names:
        predict_single_image(model, os.path.join(test_image_folder, image_name),
                             keypoint_output_folder, mask_output_folder,
                             scale_to_original_size=True)
        
    calculate_metrics(GROUND_TRUTH_FOLDER, log_folder)


if __name__ == '__main__':
    main(INFERENCE_MODEL_WEIGHTS, BEST_ARCHITECTURE, "simple_unet+_separated")