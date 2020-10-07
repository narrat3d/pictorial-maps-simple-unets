import os
import json
import math
import random
import shutil
from datetime import datetime
from timeit import default_timer as timer
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage.filters as fi
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Add, BatchNormalization, Activation
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K, layers, models, utils
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.optimizers import RMSprop
from keras_applications.resnet50 import ResNet50

from config import NUMBER_OF_BODY_PARTS, NUMBER_OF_KEYPOINTS, MASK_CHANNEL,\
    NUMBER_OF_CHANNELS, KEYPOINT_INDEX, MIRRORED_CHANNELS, mkdir_if_not_exists
import evaluation
import coco_metrics


MODEL_NAME = "bodyparts_res_unet.hdf5"

DEBUG = False
MIRROR_IMAGES = True

# input height and width for images into the network
IMAGE_SIZE = 128
# e.g. factor of 2 leads to predicted masks of size 128px
MASK_DOWNSAMPLING_FACTOR = 2
# standard deviation of gaussian kernel for creating keypoint heatmaps
SIGMA = 2


if (DEBUG):
    batch_size = 10
    epochs = 300
else:
    batch_size = 20
    epochs = 20

colors_np = np.array([
    (255, 255, 255),
    (255, 215, 0), # torso
    (255, 119, 255), # head
    (191, 255, 0), # right arm
    (255, 0, 0), # right hand
    (176, 48, 96), # right leg
    (0, 0, 128), # right foot
    (255, 228, 196), # left leg
    (0, 191, 255), # left foot
    (0, 100, 0), # left arm
    (0, 0, 0) # left hand
])


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


def conv(filters, strides, name, x):
    y = Conv2D(filters, (3, 3), strides, padding='same', name=name, 
               kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
        
    return y


def deconv_add(filters, stage, x, y):
    x = Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', 
                        name="block%s_deconv" % stage, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)  
    x = Activation('relu')(x)
    
    z = Add(name="block%s_add" % stage)([x, y])
    z = Activation('relu')(z)

    return z


def create_res_u_net_model():
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=image_shape, 
                     backend = K, layers=layers, models=models, utils=utils)
    
    y = conv(64, (1, 1), "image_conv1", model.input)
    y = conv(128, (2, 2), "image_downsampled", y)    
    y = conv(128, (1, 1), "image_conv2", y)
    
    x = model.get_layer("activation_39").output # 16px  
    x = deconv_add(512, 4, x, model.get_layer("activation_21").output) # 32px
    x = deconv_add(256, 3, x, model.get_layer("activation_9").output) # 64px
    x = deconv_add(128, 2, x, y) # 128px

    keypoints_detected = Conv2D(NUMBER_OF_CHANNELS, kernel_size=(1, 1), padding="same", 
                                activation='sigmoid', name="keypoints")(x)
    
    return Model(inputs=model.input, outputs=keypoints_detected)


def custom_metric(ytrue, ypred):    
    d = (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue[..., 0:KEYPOINT_INDEX], ypred[..., 0:KEYPOINT_INDEX])
    e = categorical_crossentropy(ytrue[..., KEYPOINT_INDEX:NUMBER_OF_CHANNELS], ypred[..., KEYPOINT_INDEX:NUMBER_OF_CHANNELS])
       
    return d + e
    
    
def train(model, train_image_folder, test_image_folder, log_folder):
    training_data_gen = DataGenerator(train_image_folder, batch_size, IMAGE_SIZE)
    
    model.summary()
    
    # learning rate is 0.001 by default
    model.compile(optimizer = RMSprop(), loss = custom_metric, metrics = ["mse"])
    
    
    if (DEBUG):
        from tensorflow.python.keras.utils import plot_model
        plot_model(model, to_file=log_folder + '/' + MODEL_NAME + '.png', show_shapes=True)
        
        start = timer()

        model.fit_generator(
            training_data_gen,
            epochs = epochs
        )
        
        model.save(os.path.join(log_folder, MODEL_NAME))
        
        end = timer()
        print("Elapsed time:", end - start)
        
    else:
        validation_data_gen = DataGenerator(test_image_folder, batch_size, IMAGE_SIZE)
        
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        csv_logger = CSVLogger(log_folder + '/training_%s_%s.csv' % (MODEL_NAME, current_time), ";")
    
        checkpoint = ModelCheckpoint(log_folder + "/weights.hdf5", save_weights_only=True, 
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
        
        x *= (MASK_DOWNSAMPLING_FACTOR)
        y *= (MASK_DOWNSAMPLING_FACTOR)
        
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


def predict_single_image(model, image_path, keypoint_output_folder, mask_output_folder):
    image_file_name = os.path.basename(image_path)  
    # print(image_file_name)
    
    image_name_without_ext, _ = os.path.splitext(image_file_name)
        
    image = Image.open(image_path)
    original_image_width = image.width
    
    detected_keypoints, predicted_body_parts_np, resized_image = predict(model, image)
    
    # create mask for removing the background from the predictions
    mask_file_path = image_path.replace("images", "masks").replace(".jpg", ".png")
    mask = Image.open(mask_file_path)
    mask = mask.getchannel(MASK_CHANNEL)
    binary_mask = mask.point(lambda p: ((p != 255) and 1) or 0)
    binary_mask_np = np.asarray(binary_mask, dtype=np.uint8)

    predicted_body_parts = Image.fromarray(np.uint8(predicted_body_parts_np), "L")
    scaled_detected_mask = predicted_body_parts.resize(image.size, Image.NEAREST)
    scaled_detected_mask_np = np.array(scaled_detected_mask)
    
    scaled_detected_mask_np *= binary_mask_np
    
    mask = Image.fromarray(np.uint8(scaled_detected_mask_np), "L")
    mask = mask.point(lambda p: (p == 0 and 255) or (p - 1))
    mask.save(os.path.join(mask_output_folder, image_name_without_ext + ".png"))
    
    colored_mask_np = colors_np[scaled_detected_mask_np]
    colored_mask = Image.fromarray(np.uint8(colored_mask_np), "RGB")
    colored_mask.save(os.path.join(mask_output_folder, image_name_without_ext + "_colored.png"))
    
    scaled_detected_keypoints = scale_keypoints(detected_keypoints, original_image_width / resized_image.width)

    draw = ImageDraw.Draw(image)
    
    for coords in scaled_detected_keypoints.values():
        x = coords[0]
        y = coords[1]
              
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black")
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="white")
        
    image.convert("RGB").save(os.path.join(keypoint_output_folder, image_name_without_ext + "_points.jpg"))
    
    keypoints_file_path = os.path.join(keypoint_output_folder, image_name_without_ext + ".json")
    json.dump(scaled_detected_keypoints, open(keypoints_file_path, "w"))


def calculate_metrics(test_folder, log_folder):
    [keypoints_error, body_parts_error] = evaluation.main(test_folder, log_folder)
    [keypoints_precision, body_parts_precision] = coco_metrics.main(test_folder, log_folder)
    
    metrics_string = ""
    metrics_string += "Keypoints error: %s\n" % keypoints_error
    metrics_string += "Body parts error: %s\n" % body_parts_error
    metrics_string += "\n"
    metrics_string += "Keypoints precision: %s\n" % keypoints_precision
    metrics_string += "Body parts precision: %s\n" % body_parts_precision
    
    print()
    print(metrics_string)
    metrics_file_path = os.path.join(log_folder, "metrics.txt")
    
    with open(metrics_file_path, "w") as metrics_file:
        metrics_file.write(metrics_string)
    


def main(train_folder, test_folder, log_folder):
    train_image_folder = os.path.join(train_folder, "images")
    test_image_folder = os.path.join(test_folder, "images")
    
    keypoint_output_folder = os.path.join(log_folder, "keypoints")
    mask_output_folder = os.path.join(log_folder, "masks")
    
    list(map(mkdir_if_not_exists, [log_folder, keypoint_output_folder, mask_output_folder]))
        
    model = create_res_u_net_model()
    # model.load_weights(os.path.join(log_folder, "weights.hdf5"))
    
    # model = load_model(os.path.join(LOG_FOLDER, MODEL_NAME), custom_objects={'custom_metric': custom_metric}) # 
    model = train(model, train_image_folder, test_image_folder, log_folder)
    
    if (DEBUG):
        image_folder = train_image_folder
        image_names = os.listdir(train_image_folder)[:10]      
    else :
        image_folder = test_image_folder
        image_names = os.listdir(test_image_folder)

    for image_name in image_names:
        predict_single_image(model, os.path.join(image_folder, image_name),
                             keypoint_output_folder, mask_output_folder)
    
    # needed otherwise getting layers by name will result in an error
    # as numbers are added incrementally
    K.reset_uids()
    
    # store this configuration
    path_of_this_file = os.path.abspath(__file__)
    shutil.copy(path_of_this_file, log_folder)
    
    if (not DEBUG):
        calculate_metrics(test_folder, log_folder)
    
    
if __name__ == '__main__':
    datasets = ["real"]
    
    test_image_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data\test"
    
    for dataset in datasets:
        train_image_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\data\%s" % dataset
        log_folder = r"C:\Users\sraimund\Pictorial-Maps-Simple-Res-U-Net\logs\%s_mirrored" % dataset
        
        main(train_image_folder, test_image_folder, log_folder)