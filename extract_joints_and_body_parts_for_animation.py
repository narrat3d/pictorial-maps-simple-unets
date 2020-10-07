import config
from evaluation import initialise_sessions, infer
import os
from PIL import Image, ImageDraw
import shutil
import json
from training import create_res_u_net_model, IMAGE_SIZE,\
    NUMBER_OF_BODY_PARTS, NUMBER_OF_KEYPOINTS, SIGMA, MASK_DOWNSAMPLING_FACTOR,\
    colors_np
import numpy as np
from tensorflow.python.keras.applications import imagenet_utils
import scipy.ndimage.filters as fi
import math

ankleRight = 0
kneeRight = 1
hipRight = 2
hipLeft = 3
kneeLeft = 4
ankleLeft = 5
hip = 6
thorax = 7
neck = 8
head = 9
wristRight = 10
elbowRight = 11
shoulderRight = 12
shoulderLeft = 13
elbowLeft = 14
wristLeft = 15

bones = {
    "torsoNeck": [thorax, head],
    "torso": [thorax, hip],
    "torsoLeftShoulder": [thorax, shoulderLeft],
    "leftUpperArm": [shoulderLeft, elbowLeft],
    "leftForeArm": [elbowLeft, wristLeft],
    "torsoRightShoulder": [thorax, shoulderRight],
    "rightUpperArm": [shoulderRight, elbowRight],
    "rightForeArm": [elbowRight, wristRight],
    "torsoLeftHip": [hip, hipLeft],
    "leftThigh": [hipLeft, kneeLeft],
    "leftLowerLeg": [kneeLeft, ankleLeft], 
    "torsoRightHip": [hip, hipRight],
    "rightThigh": [hipRight, kneeRight], 
    "rightLowerLeg": [kneeRight, ankleRight]
}

print (list(map(lambda el: [el[0] + 1, el[1] + 1], bones.values())))


bone_colors = {
    "torsoNeck": "red",
    "torso": "yellow",
    "torsoRightShoulder": "brown",
    "rightUpperArm": "green",
    "rightForeArm": "blue",
    "torsoLeftShoulder": "brown",
    "leftUpperArm": "green",
    "leftForeArm": "blue",
    "torsoRightHip": "brown",
    "rightThigh": "green",
    "rightLowerLeg": "blue",
    "torsoLeftHip": "brown",
    "leftThigh": "green",
    "leftLowerLeg": "blue",
};

body_part_names = {
    "1": "torso",
    "2": "torsoNeck"
}

body_part_connections = {
    "3": ["rightUpperArm", "rightForeArm"],
    "4": ["rightThigh", "rightLowerLeg"],
    "5": ["leftThigh", "leftLowerLeg"],
    "6": ["leftUpperArm", "leftForeArm"]             
}


def pad_image(image):
    if (image.width > image.height):
        length = image.width
        x_offset = 0
        y_offset = round((length - image.height) / 2)
    else :
        length = image.height
        x_offset = round((length - image.width) / 2)
        y_offset = 0
        
    padded_image = Image.new("RGB", (length, length), (255, 255, 255))
    padded_image.paste(image, (x_offset, y_offset))
    
    return padded_image, [x_offset, y_offset]
    

def crop_and_mask(image, bbox, mask, alpha=False):
    image_crop = image.crop(bbox)
    mask_crop = mask.crop(bbox)
    
    if (alpha):
        image_crop_masked = Image.new("RGBA", image_crop.size)
    else :
        image_crop_masked = Image.new("RGB", image_crop.size, (255, 255, 255))
        
    image_crop_masked.paste(image_crop, mask_crop)
    
    return image_crop_masked


def split_arms_and_legs(mask, keypoints, connections, masks = {}):
    mid_points = []
    
    for connection in connections:
        j = connection[0]
        k = connection[1]
        part = connection[2]
        
        masks[part] = Image.new("L", mask.size)
        
        first_keypoint = keypoints.get(j)
        second_keypoint = keypoints.get(k)
        
        if (first_keypoint == None and second_keypoint != None):
            x = second_keypoint[0]
            y = second_keypoint[1]
            
        elif (first_keypoint != None and second_keypoint == None):
            x = first_keypoint[0]
            y = first_keypoint[1]
       
        elif (first_keypoint != None and second_keypoint != None):
            # this should be the normal case
            x = (first_keypoint[0] + second_keypoint[0]) / 2
            y = (first_keypoint[1] + second_keypoint[1]) / 2
            
        else :
            continue
        
        mid_points.append([x, y, part])
    
    if (len(mid_points) == 0):
        return masks
    
    for y in range(mask.height):
        for x in range(mask.width):
            value = mask.getpixel((x, y))
            
            if (value == 0):
                continue
            
            dmin = math.hypot(mask.width, mask.height)
            j = -1
            
            for i in range(len(mid_points)):
                d = math.hypot(mid_points[i][0] - x, mid_points[i][1]-y)
                if d < dmin:
                    dmin = d
                    j = i
                    
            part = mid_points[j][2] 
            masks[part].putpixel((x, y), 255)

    return masks


def extract_characters_from_map(image, bounding_boxes, masks):
    top_left_corners = []
    paddings = []
    detected_characters = []
    
    image_with_cropped_areas = image.copy()
    white_area = Image.new("RGB", image.size, (255, 255, 255))

    for bounding_box, mask in zip(bounding_boxes, masks):
        image_mask = Image.fromarray(mask * 255, "L")
        # bounding boxes have been already resized to the actual image size
        image_mask_resized = image_mask.resize(image.size)
        
        image_crop_masked = crop_and_mask(image, bounding_box, image_mask_resized)
        image_crop_masked_padded, padding = pad_image(image_crop_masked)
        
        detected_characters.append(image_crop_masked_padded)
        top_left_corners.append([bounding_box[0], bounding_box[1]])
        paddings.append(padding)
        
        image_with_cropped_areas.paste(white_area, image_mask_resized)

    return detected_characters, top_left_corners, paddings, image_with_cropped_areas


def extract_body_parts_and_keypoints(model, image):
    body_part_masks = []
    keypoints = {}
    
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    
    image_np = np.asarray(resized_image, dtype=np.float32)
    image_np = image_np[:, :, 0:3]
    image_np = imagenet_utils.preprocess_input(image_np)
    image_np = np.expand_dims(image_np, axis=0)
    
    result = model.predict(image_np)
    first_result = result[0]

    for i in range(NUMBER_OF_BODY_PARTS + 1, NUMBER_OF_KEYPOINTS + NUMBER_OF_BODY_PARTS + 1):
        squeezed_mask = first_result[:, :, i].squeeze()

        if (squeezed_mask.max() < 0.01):
            continue
        
        detected_keypoint = fi.gaussian_filter(squeezed_mask, SIGMA, truncate=11)
        max_index = detected_keypoint.argmax()
        y,x = np.unravel_index(max_index, detected_keypoint.shape)
        
        # width = height
        image_resizing_factor = image.width / IMAGE_SIZE 
        
        x *= (MASK_DOWNSAMPLING_FACTOR * image_resizing_factor)
        y *= (MASK_DOWNSAMPLING_FACTOR * image_resizing_factor)
        
        keypoints[i - NUMBER_OF_BODY_PARTS - 1] = [int(x), int(y)]

    predicted_body_part = np.argmax(first_result[:, :, 0:1+NUMBER_OF_BODY_PARTS], axis=2)

    for i in range(NUMBER_OF_BODY_PARTS + 1):
        body_part_mask = (predicted_body_part == i).astype(int)
        body_part_mask_image = Image.fromarray(np.uint8(body_part_mask * 255), "L")
        body_part_mask_image = body_part_mask_image.resize(image.size, Image.NEAREST)
        body_part_masks.append(body_part_mask_image)

    return body_part_masks, keypoints


def visualize_keypoints_and_bones(image, keypoints):
    image_with_keypoints = image.copy()
    draw = ImageDraw.Draw(image_with_keypoints)
    
    for keypoint_coords in keypoints.values():
        x = keypoint_coords[0]
        y = keypoint_coords[1]
        
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="white")
        
    for bone_name, bone_keypoints in bones.items():
        first_keypoint = keypoints.get(bone_keypoints[0])
        second_keypoint = keypoints.get(bone_keypoints[1])
        
        if (first_keypoint != None and second_keypoint != None):
            bone_color = bone_colors[bone_name]
            draw.line([tuple(first_keypoint), tuple(second_keypoint)], fill=bone_color, width=1)
        
    return image_with_keypoints


def visualize_body_parts(image, body_part_masks):
    body_parts = Image.new("RGB", image.size)
    
    for i in range(NUMBER_OF_BODY_PARTS + 1):
        color = colors_np[i].item()
        mask_layer = Image.new("RGB", image.size, color)
        body_parts.paste(mask_layer, body_part_masks[i])
        
    return body_parts


def res_u_net_results(character_folder, res_u_net_model):
    print(character_folder)
    character_image = Image.open(os.path.join(character_folder, "image.png"))

    body_part_masks, keypoints = extract_body_parts_and_keypoints(res_u_net_model, character_image)
    # body_part_masks, keypoints = dummy_inference(character_folder)

    keypoints_json_path = os.path.join(character_folder, "keypoints.json")
    json.dump(keypoints, open(keypoints_json_path, "w"))
    
    image_with_keypoints = visualize_keypoints_and_bones(character_image, keypoints)
    image_with_keypoints.save(os.path.join(character_folder, "images_with_keypoints.png"))
    
    body_parts = visualize_body_parts(character_image, body_part_masks)
    body_parts.save(os.path.join(character_folder, "mask.png"))

    body_part_masks_by_name = {}

    for body_part_index, body_part_name in body_part_names.items():
        body_part_mask = body_part_masks[int(body_part_index)]
        body_part_masks_by_name[body_part_name] = body_part_mask
        
    for body_part_index, bone_names in body_part_connections.items():
        connections = []
        
        for bone_name in bone_names:
            bone_joints = bones[bone_name]
            connections.append(bone_joints + [bone_name])
            
        body_part_mask = body_part_masks[int(body_part_index)]
        split_arms_and_legs(body_part_mask, keypoints, connections, body_part_masks_by_name)
    
    body_part_images_by_name = {}
    body_part_images_information = {}
    
    for body_part_name, mask in body_part_masks_by_name.items():
        # mask.save(os.path.join(character_folder, "tmp_" + body_part_name + ".png"))
        
        bbox = mask.getbbox()
        
        if (bbox == None):
            continue

        body_part_images_by_name[body_part_name] = crop_and_mask(character_image, bbox, mask, alpha=True)
        
        body_part_images_information[body_part_name] = {
            "midpoint": [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1]
        }

    extracted_bones = {}

    for bone_name, bone_joints in bones.items():
        joint1 = keypoints.get(bone_joints[0])
        joint2 = keypoints.get(bone_joints[1])
        
        if (joint1 == None or joint2 == None):
            continue
        
        dx = joint2[0] - joint1[0]
        dy = joint2[1] - joint1[1]
        
        length = math.hypot(dx, dy)
        orientation = math.atan2(-dy, dx) * 180 / math.pi
        
        extracted_bones[bone_name] = {
            "orientation": orientation,
            "length": length
        }
        
        body_part_information = body_part_images_information.get(bone_name)
        
        if (body_part_information) :
            body_part_midpoint = body_part_information["midpoint"]
            mid_point_x = joint1[0] + dx/2
            mid_point_y = joint1[1] + dy/2

            extracted_bones[bone_name]["image"] = {
                "offset": {
                    "x": body_part_midpoint[0] - mid_point_x,
                    "y": body_part_midpoint[1] - mid_point_y
                },
                "width": body_part_information["width"],
                "height": body_part_information["height"],
            }
        

    for body_part_name, body_part_image in body_part_images_by_name.items():
        body_part_image.save(os.path.join(character_folder, "%s.png" % body_part_name))
    
    bones_json_path = os.path.join(character_folder, "bones.json")
    json.dump(extracted_bones, open(bones_json_path, "w"))
    
    
def dummy_inference(input_folder, image):
    mask = Image.open(os.path.join(input_folder, "mask.png"))
    keypoints = json.load(open(os.path.join(input_folder, "keypoints.json")))
    
    mask = mask.getchannel(1) 
    
    mask_np = np.zeros((mask.height, mask.width, NUMBER_OF_BODY_PARTS + 1), dtype=np.float32)
    
    for x in range(mask.width):
        for y in range(mask.height):
            body_part = mask.getpixel((x, y))
            
            if (body_part in [2, 3]):
                body_part = 3
            elif (body_part in [4, 5]):
                body_part = 4
            elif (body_part in [6, 7]):
                body_part = 5
            elif (body_part in [8, 9]):
                body_part = 6
            
            elif (body_part == 255):
                body_part = 0
            else : # [0, 1]                           
                body_part += 1
            
            mask_np[y,x,body_part] = 1.0
            
    body_part_masks = []
    
    for i in range(NUMBER_OF_BODY_PARTS + 1):
        squeezed_mask = mask_np[:,:,i].squeeze()
        squeezed_mask_image = Image.fromarray(np.uint8(squeezed_mask * 255), "L")
        body_part_masks.append(squeezed_mask_image)
    
    for key in list(keypoints.keys()):
        keypoints[int(key)] = keypoints[key]
        del keypoints[key]
    
    return body_part_masks, keypoints


def split_arms_and_legs_test():
    image = Image.open(r"E:\CNN\logs\body_parts\results\m0\char10\r7486_leg.png")
    character_folder = r"E:\CNN\logs\body_parts\results\m0\char10"
    keypoints_json_path = os.path.join(character_folder, "keypoints.json")
    keypoints = json.load(open(keypoints_json_path))
    connections = [[2, 1, "upper"], [1, 0, "lower"]]
    masks = split_arms_and_legs(image, keypoints, connections)
    masks["upper"].save(r"E:\CNN\logs\body_parts\results\m0\char10\r7486_upper_leg.png")
    masks["lower"].save(r"E:\CNN\logs\body_parts\results\m0\char10\r7486_lower_leg.png")


if __name__ == '__main__':
    res_u_net_model_folder = r"E:\CNN\logs\body_parts\mixed"
    res_u_net_model = create_res_u_net_model()
    res_u_net_model.load_weights(os.path.join(res_u_net_model_folder, "weights.hdf5"))
    """
    character_folder = r"E:\CNN\logs\body_parts\results\0c9a7642c1b9554f22feba75c8069e1f--maps-maps-maps-maps-posters\char0"
    res_u_net_results(character_folder, res_u_net_model)
    """
    output_folder = r"E:\CNN\logs\body_parts\results"
    test_images_folder = r"E:\CNN\masks\data\character_maps_separated\test"
    # test_images_folder = r"E:\CNN\masks\data\character_maps_mixed\eval\images"
    test_image_names = os.listdir(test_images_folder)[:20]
    
    # mask_rcnn_model_folder_path = os.path.join(config.LOG_FOLDER, "1st_separated_with_objects_run_stride8_0.125_0.25_0.5_1.0")
    inference_model_path = os.path.join(config.LOG_FOLDER, "inference-2000", "frozen_inference_graph.pb")
    # inference_model_path = r"E:\CNN\models\mask_rcnn_resnet101_atrous_coco_2018_01_28\frozen_inference_graph.pb"
    
    (image_session, image_tensor, image_placeholder,
      detection_session, detection_tensor, detection_placeholder) = initialise_sessions(inference_model_path)
      
    for image_name in test_image_names:
        image_file_path = os.path.join(test_images_folder, image_name)
        
        (detection_bounding_boxes, detection_masks, image) = \
            infer(image_session, image_tensor, image_placeholder, 
                  detection_session, detection_tensor, detection_placeholder, 
                  image_file_path) 
            
        character_images, top_left_corners, paddings, image_with_cropped_areas = \
            extract_characters_from_map(image, detection_bounding_boxes, detection_masks)
        
        output_map_folder = os.path.join(output_folder, os.path.splitext(image_name)[0])
        
        if (os.path.exists(output_map_folder)):
            shutil.rmtree(output_map_folder)
        
        os.mkdir(output_map_folder)
        image.save(os.path.join(output_map_folder, "original_map.png"))
        image_with_cropped_areas.save(os.path.join(output_map_folder, "extracted_map.png"))
        
        counter = 0
        
        for character_image, top_left_corner, padding in zip(character_images, top_left_corners, paddings):
            character_output_folder = os.path.join(output_map_folder, "char%s" % counter)
            os.mkdir(character_output_folder)
            
            character_image.save(os.path.join(character_output_folder, "image.png"))
            
            top_left_corner_json_path = os.path.join(character_output_folder, "topLeftCorner.json")
            json.dump(top_left_corner, open(top_left_corner_json_path, "w"))
            
            padding_json_path = os.path.join(character_output_folder, "padding.json")
            json.dump(padding, open(padding_json_path, "w"))
            
            res_u_net_results(character_output_folder, res_u_net_model)
            
            counter += 1
            
        map_config = {
            "characterNumber": counter,
            "width": image.width,
            "height": image.height
        }
        map_config_file_path = os.path.join(output_map_folder, "config.json")
        json.dump(map_config, open(map_config_file_path, "w"))
    