'''
input:
training and test data (images, masks, keypoints)

output:
weights and evaluation metrics for the best model of a run
visualization of the model
detected body part masks and pose keypoints
intermediate evaluation metrics

purpose:
train and evaluate different architectures and datasets 
'''
import os
import json
import math
import random
import shutil
from timeit import default_timer as timer
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage.filters as fi
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Add, BatchNormalization, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K, layers, models, utils
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.optimizers import RMSprop
from keras_applications.resnet50 import ResNet50

from config import WEIGHTS_FILE_NAME, CSV_LOG_FILE_NAME, PLOT_LOG_FILE_NAME, \
    METRICS_LOG_FILE_NAME, NUMBER_OF_BODY_PARTS, NUMBER_OF_KEYPOINTS, MASK_CHANNEL,\
    NUMBER_OF_CHANNELS, KEYPOINT_INDEX, MIRRORED_CHANNELS, mkdir_if_not_exists,\
    DATASET_FOLDER, LOG_FOLDER, RUNS, DATASETS, ARCHITECTURES,\
    DEBUG, IMAGE_SIZE, MIRROR_IMAGES, MASK_DOWNSAMPLING_FACTOR, SIGMA, COLORS,\
    BONES, GROUND_TRUTH_FOLDER
import error_metrics
import coco_metrics


if (DEBUG):
    batch_size = 10
    epochs = 300
else:
    batch_size = 15
    epochs = 15

colors_np = np.array(COLORS)


#source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, image_dir, batch_size, img_size):
        image_names = os.listdir(image_dir)
        
        if (DEBUG):
            image_names = image_names[:batch_size]
            
        image_names.sort()
        
        image_paths = list(map(lambda image_name: os.path.join(image_dir, image_name), image_names))
        
        self.file_names = image_paths
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.on_epoch_end()

    
    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        file_names_temp = [self.file_names[k] for k in indexes]
        X, Y = self.__data_generation(file_names_temp)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_names))
        np.random.shuffle(self.indexes)


    def __data_generation(self, image_file_paths):
        mask_size = int(self.img_size / MASK_DOWNSAMPLING_FACTOR)
        image_size = int(self.img_size)
        
        source = np.empty((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        target = np.zeros((self.batch_size, mask_size, mask_size, NUMBER_OF_CHANNELS), dtype=np.float32)

        for i, image_file_path in enumerate(image_file_paths):
            image = Image.open(image_file_path)
            resized_image = image.resize((image_size, image_size), Image.NEAREST)
            resized_image_np = np.asarray(resized_image, dtype=np.float32)
            
            image_np = resized_image_np[:, :, 0:3]
            image_np = imagenet_utils.preprocess_input(image_np)
                        
            mask_file_path = image_file_path.replace("images", "masks").replace(".jpg", ".png")
            mask = Image.open(mask_file_path)
            mask = mask.getchannel(MASK_CHANNEL)
            
            resized_mask = mask.resize((mask_size, mask_size), Image.NEAREST)
                            
            reclassified_mask = resized_mask.point(lambda p: (p != 255 and (p + 1)) or 0)
            reclassified_mask_np = np.asarray(reclassified_mask, dtype=np.int32)
            
            # source: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
            body_parts_mask_np = np.eye(NUMBER_OF_BODY_PARTS + 1, dtype=np.float32)[reclassified_mask_np]
            
            keypoint_mask = np.zeros((mask_size, mask_size, NUMBER_OF_KEYPOINTS + 1), dtype=np.float32) 


            keypoint_file_path = image_file_path.replace("images", "keypoints").replace(".jpg", ".json")
            keypoints = json.load(open(keypoint_file_path))
                
            for keypoint_id, keypoint in keypoints.items():  
                keypoint_id = int(keypoint_id)
                                  
                keypoint_x = math.floor(keypoint[0] / mask.width * mask_size)
                keypoint_y = math.floor(keypoint[1] / mask.height * mask_size) 

                if (keypoint_x < 0 or keypoint_x >= mask_size or keypoint_y < 0 or keypoint_y >= mask_size):
                    continue
                
                # convert keypoints to 2D gaussian distribution heat maps
                joint_heat_map = np.zeros((mask_size, mask_size), dtype=np.float32) 
                joint_heat_map[keypoint_y, keypoint_x] = 1
                joint_heat_map = fi.gaussian_filter(joint_heat_map, SIGMA, truncate=11)
                joint_heat_map /= joint_heat_map[keypoint_y, keypoint_x]

                keypoint_mask[:, :, keypoint_id] = joint_heat_map

            # last keypoint channel is the difference from the sum of all keypoint heatmaps
            keypoint_summed = np.minimum(np.sum(keypoint_mask[:, :, 0:NUMBER_OF_KEYPOINTS], axis=2), 
                                         np.ones((mask_size, mask_size), dtype=np.float32))
                            
            keypoint_mask[:, :, NUMBER_OF_KEYPOINTS] = np.ones((mask_size, mask_size), dtype=np.float32) - keypoint_summed
                        
            stacked_masks = np.concatenate([body_parts_mask_np, keypoint_mask], axis=2)
            
            mirror_image = random.choice([True, False])
            
            if (MIRROR_IMAGES and mirror_image and not DEBUG):                
                image_np = np.fliplr(image_np)                
                stacked_masks = np.fliplr(stacked_masks)
                
                # swap left and right parts
                stacked_masks = stacked_masks[..., MIRRORED_CHANNELS]
            
            source[i,] = image_np
            target[i,] = stacked_masks

        return source, target


def conv(filters, strides, name, x, kernel_size=(3, 3)):
    y = Conv2D(filters, kernel_size, strides, padding='same', name=name, 
               kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
        
    return y


def deconv(filters, stage, x):
    x = Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', 
                        name="block%s_deconv" % stage, kernel_initializer='he_normal')(x)
    
    x = BatchNormalization()(x)  
    x = Activation('relu')(x)

    return x


def deconv_add(filters, stage, x, *y):
    x = deconv(filters, stage, x)
    
    xy = [x] + list(y)
    
    z = Add(name="block%s_add" % stage)(xy)
    z = Activation('relu')(z)

    return z


def create_model(architecture):
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=image_shape, 
                     backend = K, layers=layers, models=models, utils=utils)
    
    x00_raw = model.get_layer("activation").output
    x00 = conv(128, (1, 1), "first_conv", x00_raw)
    
    x10 = model.get_layer("activation_9").output
    x20 = model.get_layer("activation_21").output
    x30 = model.get_layer("activation_39").output


    if (architecture == "simple_deconv"):
        x21 = deconv(512, 21, x30)
        x12 = deconv(256, 12, x21)
        x03 = deconv(128, 3, x12)
    
    elif (architecture == "simple_unet"):
        x21 = deconv_add(512, 21, x30, x20)
        x12 = deconv_add(256, 12, x21, x10)
        x03 = deconv_add(128, 3, x12, x00)
        
    elif (architecture == "simple_unet+"):
        x21 = deconv_add(512, 21, x30, x20)
        x11 = deconv_add(256, 11, x20, x10)
        x12 = deconv_add(256, 12, x21, x11)
        x01 = deconv_add(128, 1, x10, x00)
        x02 = deconv_add(128, 2, x11, x01)
        x03 = deconv_add(128, 3, x12, x02)
    
    elif (architecture == "simple_unet++"):
        x21 = deconv_add(512, 21, x30, x20)
        x11 = deconv_add(256, 11, x20, x10)
        x12 = deconv_add(256, 12, x21, x11, x10)
        x01 = deconv_add(128, 1, x10, x00)
        x02 = deconv_add(128, 2, x11, x01, x00)
        x03 = deconv_add(128, 3, x12, x02, x01, x00)
    
    final_conv = Conv2D(NUMBER_OF_CHANNELS, kernel_size=(1, 1), padding="same", 
                        activation='sigmoid', name="final_conv")(x03) 
    
    return Model(inputs=model.input, outputs=final_conv)


def custom_metric(ytrue, ypred):    
    d = (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue[..., 0:KEYPOINT_INDEX], ypred[..., 0:KEYPOINT_INDEX])
    e = categorical_crossentropy(ytrue[..., KEYPOINT_INDEX:NUMBER_OF_CHANNELS], ypred[..., KEYPOINT_INDEX:NUMBER_OF_CHANNELS])
       
    return d + e

    
def train(model, train_image_folder, test_image_folder, log_folder):
    training_data_gen = DataGenerator(train_image_folder, batch_size, IMAGE_SIZE)
    
    model.summary()
    
    # learning rate is 0.001 by default
    model.compile(optimizer = RMSprop(), loss = custom_metric, metrics = ["mse"])
    
    from tensorflow.python.keras.utils import plot_model
    plot_file_path = os.path.join(log_folder, PLOT_LOG_FILE_NAME)
    plot_model(model, to_file=plot_file_path, show_shapes=True)
    
    weights_file_path = os.path.join(log_folder, WEIGHTS_FILE_NAME)
    
    if (DEBUG):        
        start = timer()

        model.fit_generator(
            training_data_gen,
            epochs = epochs
        )
        
        model.save_weights(weights_file_path)
        
        end = timer()
        print("Elapsed time:", end - start)
        
    else:
        validation_data_gen = DataGenerator(test_image_folder, batch_size, IMAGE_SIZE)
        
        csv_log_file_path = os.path.join(log_folder, CSV_LOG_FILE_NAME)
        csv_logger = CSVLogger(csv_log_file_path, ";")
    
        checkpoint = ModelCheckpoint(weights_file_path, save_weights_only=True, 
                                     monitor='val_mean_squared_error', mode="min", verbose=1, save_best_only=True, period=1)
        
        model.fit_generator(
            training_data_gen,
            epochs = epochs,
            validation_data=validation_data_gen,
            callbacks=[csv_logger, checkpoint]
        )
    
    return model


def predict(model, image):
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    
    image_np = np.asarray(resized_image, dtype=np.float32)
    image_np = image_np[:, :, 0:3]
    image_np = imagenet_utils.preprocess_input(image_np)
    image_np = np.expand_dims(image_np, axis=0)
    
    result = model.predict(image_np)

    predicted_body_parts = np.argmax(result[0][:, :, 0:KEYPOINT_INDEX], axis=2)

    detected_keypoints = {}

    for i in range(KEYPOINT_INDEX, NUMBER_OF_CHANNELS - 1):
        squeezed_mask = result[0][:, :, i].squeeze()
    
        if (squeezed_mask.max() < 0.01):
            continue
        
        detected_keypoint = fi.gaussian_filter(squeezed_mask, SIGMA, truncate=11)
        max_index = detected_keypoint.argmax()
        y,x = np.unravel_index(max_index, detected_keypoint.shape)
        
        detected_keypoints[i - KEYPOINT_INDEX] = [x, y]

    return detected_keypoints, predicted_body_parts, resized_image


def scale_keypoints(keypoints, scale):
    scaled_keypoints = {}
    
    for keypoint_index, keypoint_coords in keypoints.items():
        scaled_keypoints[keypoint_index] = [
            int(keypoint_coords[0] * scale),
            int(keypoint_coords[1] * scale),
        ]
        
    return scaled_keypoints


def predict_single_image(model, image_path, keypoint_output_folder, mask_output_folder, scale_to_original_size=False):
    image_file_name = os.path.basename(image_path)  
    image_name_without_ext, _ = os.path.splitext(image_file_name)
        
    image = Image.open(image_path)
    original_image_width = image.width
    
    detected_keypoints, predicted_body_parts_np, resized_image = predict(model, image)
    
    # create mask for removing the background from the predictions
    mask_file_path = image_path.replace("images", "masks").replace(".jpg", ".png")
    mask = Image.open(mask_file_path)
    mask = mask.getchannel(MASK_CHANNEL)
    binary_mask = mask.point(lambda p: ((p != 255) and 1) or 0)
    
    if (not scale_to_original_size):
        binary_mask = binary_mask.resize(resized_image.size, Image.NEAREST)
        
    binary_mask_np = np.asarray(binary_mask, dtype=np.uint8)

    predicted_body_parts = Image.fromarray(np.uint8(predicted_body_parts_np), "L")
    
    if (scale_to_original_size):
        mask_size = image.size 
    else:
        mask_size = resized_image.size
    scaled_detected_mask = predicted_body_parts.resize(mask_size, Image.NEAREST)
    scaled_detected_mask_np = np.array(scaled_detected_mask)
    
    scaled_detected_mask_np *= binary_mask_np
    
    mask = Image.fromarray(np.uint8(scaled_detected_mask_np), "L")
    mask = mask.point(lambda p: (p == 0 and 255) or (p - 1))
    mask.save(os.path.join(mask_output_folder, image_name_without_ext + ".png"))
    
    colored_mask_np = colors_np[scaled_detected_mask_np]
    colored_mask = Image.fromarray(np.uint8(colored_mask_np), "RGB")
    colored_mask.save(os.path.join(mask_output_folder, image_name_without_ext + "_colored.png"))
    
    if (scale_to_original_size):
        keypoint_scale_factor = original_image_width / (resized_image.width / MASK_DOWNSAMPLING_FACTOR)
        image_with_skeleton = image
    else:
        keypoint_scale_factor = MASK_DOWNSAMPLING_FACTOR
        image_with_skeleton = resized_image
    
    scaled_detected_keypoints = scale_keypoints(detected_keypoints, keypoint_scale_factor)  

    draw = ImageDraw.Draw(image_with_skeleton)
 
    for bone in BONES:
        start_keypoint_id = bone[0]
        end_keypoint_id = bone[1]
        color = bone[2]
        
        if (not start_keypoint_id in scaled_detected_keypoints or 
            not end_keypoint_id in scaled_detected_keypoints):
            continue
        
        start_point = tuple(scaled_detected_keypoints[start_keypoint_id])
        end_point = tuple(scaled_detected_keypoints[end_keypoint_id])
        
        draw.line([start_point, end_point], fill=color, width=4)

    for coords in scaled_detected_keypoints.values():
        x = coords[0]
        y = coords[1]
              
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="black")
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="white")
 
    image_with_skeleton_path = os.path.join(keypoint_output_folder, image_name_without_ext + "_skeleton.jpg")
    image_with_skeleton.convert("RGB").save(image_with_skeleton_path)
    
    keypoints_file_path = os.path.join(keypoint_output_folder, image_name_without_ext + ".json")
    json.dump(scaled_detected_keypoints, open(keypoints_file_path, "w"))


def calculate_metrics(test_folder, log_folder):
    [keypoints_error, body_parts_error] = error_metrics.main(test_folder, log_folder)
    [keypoints_precisions, body_parts_precisions] = coco_metrics.main(test_folder, log_folder)
    
    metrics = {
        "error": {},
        "coco": {}
    }
    
    metrics["error"]["keypoints"] = keypoints_error
    metrics["error"]["bodyparts"] = body_parts_error
    metrics["coco"]["keypoints"] = keypoints_precisions
    metrics["coco"]["bodyparts"] = body_parts_precisions
    
    print()
    print(metrics)
    metrics_file_path = os.path.join(log_folder, METRICS_LOG_FILE_NAME)
    
    with open(metrics_file_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)


def main(train_folder, log_folder, architecture):
    train_image_folder = os.path.join(train_folder, "images")
    test_image_folder = os.path.join(GROUND_TRUTH_FOLDER, "images")
    
    keypoint_output_folder = os.path.join(log_folder, "keypoints")
    mask_output_folder = os.path.join(log_folder, "masks")
    
    list(map(mkdir_if_not_exists, [log_folder, keypoint_output_folder, mask_output_folder]))
        
    model = create_model(architecture)
    
    model = train(model, train_image_folder, test_image_folder, log_folder)
    model.load_weights(os.path.join(log_folder, WEIGHTS_FILE_NAME))
    
    if (DEBUG):
        image_folder = train_image_folder
        image_names = os.listdir(train_image_folder)[:10]      
    else :
        image_folder = test_image_folder
        image_names = os.listdir(test_image_folder)

    for image_name in image_names:
        predict_single_image(model, os.path.join(image_folder, image_name),
                             keypoint_output_folder, mask_output_folder,
                             scale_to_original_size=True)
    
    # needed otherwise getting layers by name will result in an error
    # as numbers are added incrementally
    K.reset_uids()
    
    # store this configuration
    path_of_this_file = os.path.abspath(__file__)
    shutil.copy(path_of_this_file, log_folder)
    
    if (not DEBUG):
        calculate_metrics(GROUND_TRUTH_FOLDER, log_folder)
    
    
if __name__ == '__main__':
    for architecture in ARCHITECTURES:
        for dataset in DATASETS:
            train_folder = os.path.join(DATASET_FOLDER, dataset)
            
            for run in RUNS:
                results_folder = os.path.join(LOG_FOLDER, architecture)
                mkdir_if_not_exists(results_folder)
                log_folder = os.path.join(results_folder, "%s_%s" % (dataset, run))
                
                main(train_folder, log_folder, architecture)