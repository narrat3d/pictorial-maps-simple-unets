from timeit import default_timer as timer
from PIL import Image, ImageDraw
import os
from time import gmtime, strftime
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Add, BatchNormalization, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.utils import Sequence
import json
import scipy.ndimage.filters as fi
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K, layers, models, utils
from keras_applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import imagenet_utils
import math
from tensorflow.python.keras.optimizers import RMSprop
import random
from datetime import datetime
from helper_functions import scale_keypoints, mkdir_if_not_exists

# todo: argmax loss/metrics for body parts

MASK_CHANNEL = 0
TRAIN_IMAGE_FOLDER = r"E:\CNN\masks\data\figures\real\images"
TEST_IMAGE_FOLDER = r"E:\CNN\masks\data\figures\test\images"

MODEL_NAME = "bodyparts_res_unet.hdf5"

LOG_FOLDER = r"E:\CNN\logs\body_parts\real"

keypoint_output_folder = os.path.join(LOG_FOLDER, "keypoints")
mask_output_folder = os.path.join(LOG_FOLDER, "masks")

list(map(mkdir_if_not_exists, [keypoint_output_folder, mask_output_folder]))

MIRRORED_BODY_PARTS = [0, 1, 5, 4, 3, 2]
MIRRORED_JOINTS = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

mirrored_body_parts = list(map(lambda x: x + 1, MIRRORED_BODY_PARTS))
mirrored_joints = list(map(lambda x: x + 1 + len(mirrored_body_parts), MIRRORED_JOINTS))

mirrored_masks = [0] + mirrored_body_parts + mirrored_joints

cache = {}

NUMBER_OF_KEYPOINTS = 16
NUMBER_OF_BODY_PARTS = 6
keypoint_index = 1 + NUMBER_OF_BODY_PARTS
output_channels = 1 + NUMBER_OF_BODY_PARTS + NUMBER_OF_KEYPOINTS + 1

SIGMA = 2
MASK_DOWNSAMPLING_FACTOR = 2
IMAGE_SIZE = 256

debug = True

if (debug):
    batch_size = 10
    epochs = 800
else:
    batch_size = 15
    epochs = 10

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

datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)


class DataGenerator(Sequence):

    def __init__(self, image_dir, batch_size, img_size):
        image_names = os.listdir(image_dir)
        
        if (debug):
            image_names = image_names[:batch_size]
        # image_names = list(filter(lambda image_name: image_name != "r10.png", image_names))
        image_names.sort()
        
        image_paths = list(map(lambda image_name: os.path.join(image_dir, image_name), image_names))
        
        # image_paths = list(filter(lambda image_path: Image.open(image_path).size[0] >= 64, image_paths))
        
        self.file_names = image_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.on_epoch_end()

    'Denotes the number of batches per epoch'
    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        file_names_temp = [self.file_names[k] for k in indexes]
        # print(file_names_temp)
        X, Y = self.__data_generation(file_names_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_names))
        np.random.shuffle(self.indexes)

    def __data_generation(self, image_file_paths):
        mask_size = int(self.img_size / MASK_DOWNSAMPLING_FACTOR)
        image_size = int(self.img_size)
        
        source = np.empty((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)

        target = np.zeros((self.batch_size, mask_size, mask_size, output_channels), dtype=np.float32)

        for i, image_file_path in enumerate(image_file_paths):
            cache_entry = cache.get(image_file_path)

            if (cache_entry is None):
                image = Image.open(image_file_path)
                resized_image = image.resize((image_size, image_size), Image.NEAREST)
                # resized_image2 = image.resize((mask_size, mask_size), Image.NEAREST) 
                # resized_image.save(os.path.join(results_dir, "test.png"))
                resized_image_np = np.asarray(resized_image, dtype=np.float32)
                
                image_np = resized_image_np[:, :, 0:3]
                image_np = imagenet_utils.preprocess_input(image_np, mode="tf")
                      
                mask_file_path = image_file_path.replace("images", "masks").replace(".jpg", ".png")
                mask = Image.open(mask_file_path)
                mask = mask.getchannel(MASK_CHANNEL)
                
                resized_mask = mask.resize((mask_size, mask_size), Image.NEAREST)
                                
                # image_np_with_alpha = np.concatenate((image_np, binary_mask_np))
                
                reclassified_mask = resized_mask.point(lambda p: (p != 255 and (p + 1)) or 0)
                
                resized_mask_np = np.asarray(reclassified_mask, dtype=np.int32)
                
                # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
                mask_np = np.eye(NUMBER_OF_BODY_PARTS + 1)[resized_mask_np]
                
                # mask_np = np.zeros((mask_size, mask_size, NUMBER_OF_BODY_PARTS + 1), dtype=np.float32)
                keypoint_mask = np.zeros((mask_size, mask_size, NUMBER_OF_KEYPOINTS + 1), dtype=np.float32) 


                keypoint_file_path = image_file_path.replace("images", "keypoints").replace(".jpg", ".json")
                keypoints = json.load(open(keypoint_file_path))
                    
                for keypoint_id, keypoint in keypoints.items():  
                    keypoint_id = int(keypoint_id)                         
                    keypoint_x = math.floor(keypoint[0] / mask.width * mask_size)
                    keypoint_y = math.floor(keypoint[1] / mask.height * mask_size) 

                    if (keypoint_x < 0 or keypoint_x >= mask_size or keypoint_y < 0 or keypoint_y >= mask_size):
                        continue
                     
                    joint_heat_map = np.zeros((mask_size, mask_size), dtype=np.float32) 
                    joint_heat_map[keypoint_y, keypoint_x] = 1
                    joint_heat_map = fi.gaussian_filter(joint_heat_map, SIGMA, truncate=11)
                
                    joint_heat_map /= joint_heat_map[keypoint_y, keypoint_x]

                    keypoint_mask[:, :, keypoint_id] = joint_heat_map


                
                keypoint_summed = np.minimum(np.sum(keypoint_mask[:, :, 0:NUMBER_OF_KEYPOINTS], axis=2), 
                                             np.ones((mask_size, mask_size), dtype=np.float32))
                                
                keypoint_mask[:, :, NUMBER_OF_KEYPOINTS] = np.ones((mask_size, mask_size), dtype=np.float32) - keypoint_summed
                
                # keypoint_summed = np.expand_dims(keypoint_summed, axis=2)
                # keypoint_summed = np.repeat(keypoint_summed, NUMBER_OF_BODY_PARTS + 1, axis=2)
                
                # body_mask_minus_keypoint = mask_np - keypoint_summed
                # body_mask_minus_keypoint = body_mask_minus_keypoint * mask_np
                
                stacked_masks = np.concatenate([mask_np, keypoint_mask], axis=2)
                
                # numpy uses mirrored axes https://github.com/python-pillow/Pillow/issues/2619 !!!
                """
                tmp = Image.fromarray(np.swapaxes(np.sum(mask_np[:, :, keypoint_index_offset:output_channels], axis=2) * 255, 0, 1)).convert("L")
                # tmp = Image.fromarray(np.swapaxes(mask_np[:, :, 9] * 255, 0, 1)).convert("L")
                image_name = os.path.basename(image_file_path)
                
                tmp.save(r"E:\CNN\masks\keras\body_parts\logs\real/keypoints_" + image_name)
                """
                """
                # Image.fromarray(np.uint8(image_np * 255)).save(r"E:\CNN\masks\keras\body_parts\logs\real/" + image_name)
                if (image_name == "0.png"):
                    resized_mask.save(os.path.join(results_dir, "resized_mask.png"))
                    
                    for j in range(1+NUMBER_OF_BODY_PARTS):
                        tmp_mask = Image.fromarray(mask_np[:, :, j] * 255).convert("L")
                        tmp_mask.save(os.path.join(results_dir, "mask_%s_%s" % (j, image_name)))
                
                """
                               
                """
                image_name = os.path.basename(image_file_path) 
                for k in range(NUMBER_OF_BODY_PARTS + NUMBER_OF_KEYPOINTS + 1):
                    tmp_mask = Image.fromarray(np.uint8(stacked_masks[:, :, k] * 255)).convert("L")
                    tmp_mask.save(os.path.join(results_dir, "%s_mask_%s.png" % (os.path.splitext(image_name)[0], k)))
                """
                
                # Image.fromarray(np.uint8(one_hot_mask_np * 255)).save(os.path.join(LOG_FOLDER, "one_hot_mask.png"))
                
                # resized_image2.save(os.path.join(results_dir, image_name + "_resized.png"))
                # Image.fromarray(np.uint8(keypoints_mask_np[:, :, 0] * 255)).save(os.path.join(results_dir, "right_ankle_heat_map.png"))
                # Image.fromarray(np.uint8(keypoints_mask_np[:, :, 1] * 255)).save(os.path.join(results_dir, "right_knee_heat_map.png"))
                """
                cache[image_file_path] = {
                    "image": image_np,
                    # "body_mask": one_hot_mask_np,
                    # "body_parts": body_mask_minus_keypoint,
                    # "body_part_halves": body_part_halves_masks,
                    # "joints": joint_masks
                    "keypoints": stacked_masks
                }
                """
                
            else :
                pass
                # image_np = cache_entry["image"]
                # one_hot_mask_np = cache_entry["body_mask"]
                # mask_np = cache_entry["body_parts"]
                # body_part_halves_masks = cache_entry["body_part_halves"]
                # joint_masks = cache_entry["joints"]    
                # stacked_masks = cache_entry["keypoints"]       
            
            """
            mirror_image = random.choice([True, False])
            
            if (mirror_image and not debug):
                # image_name = os.path.basename(image_file_path)
                # Image.fromarray(np.uint8(image_np * 255)).save(os.path.join(LOG_FOLDER, image_name))
                
                image_np = np.fliplr(image_np)
                
                # Image.fromarray(np.uint8(image_np * 255)).save(os.path.join(LOG_FOLDER, "mirror_" + image_name))
                
                stacked_masks = np.fliplr(stacked_masks)
                # swap left and right parts
                stacked_masks = stacked_masks[..., mirrored_masks]
            """
            source[i,] = image_np # _with_alpha
            # source2[i, ] = one_hot_mask_np
            target[i,] = stacked_masks
            """
            body_parts_target[i,] = mask_np # mask_np[:, :, 1 + NUMBER_OF_BODY_PARTS: 1 + NUMBER_OF_BODY_PARTS + NUMBER_OF_KEYPOINTS]
            keypoints_target[i,] = keypoint_mask
            
            for body_part_halves_name in body_part_halves_names:
                body_part_halves_targets[body_part_halves_name][i, ] = body_part_halves_masks[body_part_halves_name]
                
            for joint_name in joint_names:
                joint_targets[joint_name][i, ] = joint_masks[joint_name]
            """
        # source = datagen.standardize(source)
        """
        targets = [body_parts_target]
        
        for body_part_halves_name in body_part_halves_names:
            targets.append(body_part_halves_targets[body_part_halves_name])
            
        for joint_name in joint_names:    
            targets.append(joint_targets[joint_name])
        """
        return source, target # [source, source2], 


def double_conv(filters, name, x):
    # x = conv_block(x, 3, [int(filters / 4), int(filters / 4), filters], name+"_conv_block", str(1), strides=(1, 1))
    # x = identity_block(x, 3, [int(filters / 4), int(filters / 4), filters], name+"_identity_block", str(1))
    
    conv1 = Conv2D(filters, (3, 3), padding='same', name='%s_conv1' % name, kernel_initializer='he_normal')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # conv2 = Conv2D(filters, (3, 3), padding='same', name='%s_conv2' % name, kernel_initializer='he_normal')(conv1) # kernel_initializer='he_normal'
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Activation('relu')(conv2)
        
    return conv1


def deconv_concat_double_conv(filters, stage, x, y):
    x = Conv2DTranspose(
        filters, (3, 3), strides=(2, 2), padding='same', name="block%s_deconv" % stage)(x) 
    x = BatchNormalization()(x)  
    x = Activation('relu')(x)
    
    z = Add(name="block%s_concat" % stage)([x, y])
    z = Activation('relu')(z)
    
    # block_concat = Add(name="block%s_concat" % stage)([x, y])
    
    # z = double_conv(filters, "block%s_up" % stage, z)
    
    # x = identity_block(x, 3, [int(filters / 4), int(filters / 4), filters], stage=stage, block='a_up')
    # x = identity_block(x, 3, [int(filters / 4), int(filters / 4), filters], stage=stage, block='b_up')
    
    # x = identity_block(block_concat, 3, [int(filters / 4), int(filters / 4), filters], stage=stage, block='b_up')

    return z


def deconv(stage, x):
    x = Conv2DTranspose(
        256, (4, 4), strides=(2, 2), padding='same', name="block%s_deconv" % stage)(x)     
    x = BatchNormalization(momentum=0.9)(x)  
    x = Activation('relu')(x)
    
    return x



def create_res_u_net_model():
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # body_mask_shape = (img_size / MASK_DOWNSAMPLING_FACTOR, img_size / MASK_DOWNSAMPLING_FACTOR)
    
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=image_shape, 
                     backend = K, layers=layers, models=models, utils=utils)
    
    image_input = model.input
    
    y = double_conv(64, "image_conv1", image_input)
    
    y = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name="image_downsampled")(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = double_conv(128, "image_conv2", y)
    
    # body_mask_input = Input(body_mask_shape, name="one_hot_mask_input", dtype=np.int64)
    
    # for layer in model.layers:
    #     layer.trainable = False
    
    x = model.get_layer("activation_39").output # model.layers[-3].output
    
    # x = deconv_concat_double_conv(1024, 5, x, model.get_layer("activation_39").output)
    
    x = deconv_concat_double_conv(512, 4, x, model.get_layer("activation_21").output) # 32px
    x = deconv_concat_double_conv(256, 3, x, model.get_layer("activation_9").output) # 64px
    x = deconv_concat_double_conv(128, 2, x, y) # 128px model.get_layer("activation").output
    
    # x = deconv(4, x)
    # x = deconv(3, x)
    # x = deconv(2, x)
    
    keypoints_detected = Conv2D(output_channels, kernel_size=(1, 1), padding="same", activation='sigmoid', 
                                name="keypoints")(x)
    
    # x = double_conv(64, "final_convs", x)
    """
    body_parts_classified = Conv2D(NUMBER_OF_BODY_PARTS + 1, kernel_size=(1, 1), padding="same", activation='sigmoid', 
                                   name="body_parts")(x)
    
    # resized_img_input = model.get_layer("activation").output
    # feature_map_input = model.get_layer("activation_9").output
    resized_img_input = Lambda(lambda x: tf.image.resize_images(x, (128, 128)), 
                               name="resized_img_input")(model.input)

    body_part_halves_classified = {}
    joints_classified = {}
    outputs = [body_parts_classified]
    
    for body_part_halves_name in body_part_halves_names:
        body_part_index = body_part_mapping[body_part_halves_name] + 1
        body_part_halves_classified[body_part_halves_name] = \
            classify_body_part_halves(body_part_halves_name, body_part_index, body_parts_classified, body_mask_input, resized_img_input)
            
        outputs.append(body_part_halves_classified[body_part_halves_name])

    # makes things a bit easier
    body_part_halves_classified["body_parts"] = body_parts_classified

    for joint_name in joint_names:
        body_part_details = full_body_part_mapping[joint_name]
        number = body_part_details["number"]
        body_part_parent = body_part_details["parent"]
        body_part_classified = body_part_halves_classified[body_part_parent]
        
        joints_classified[joint_name] = detect_joints(joint_name, number, body_part_classified, body_mask_input, resized_img_input)

        outputs.append(joints_classified[joint_name])
    """
    
    return Model(inputs=model.input, outputs=keypoints_detected)


def custom_metric(ytrue, ypred):    
    d = (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue[..., 0:keypoint_index], ypred[..., 0:keypoint_index])
    e = categorical_crossentropy(ytrue[..., keypoint_index:output_channels], ypred[..., keypoint_index:output_channels])
       
    return d + e
    
    
def train(model):
    training_data_gen = DataGenerator(TRAIN_IMAGE_FOLDER, batch_size, IMAGE_SIZE)
    validation_data_gen = DataGenerator(TEST_IMAGE_FOLDER, batch_size, IMAGE_SIZE)
    
    model.summary()
    
    # learning rate is 0.001 by default
    model.compile(optimizer = RMSprop(), loss = custom_metric, metrics = ["mse"])
    
    
    if (debug):
        from tensorflow.python.keras.utils import plot_model
        plot_model(model, to_file=LOG_FOLDER + '/' + MODEL_NAME + '.png', show_shapes=True)
        
        start = timer()
        
        model.fit_generator(
            training_data_gen,
            epochs = epochs,
        )
        model.save(os.path.join(LOG_FOLDER, MODEL_NAME))
        
        end = timer()
        print("Elapsed time:", end - start)
        
    else:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        csv_logger = CSVLogger(LOG_FOLDER + '/training_%s_%s.csv' % (MODEL_NAME, current_time))
    
        checkpoint = ModelCheckpoint(LOG_FOLDER + "/weights.hdf5", save_weights_only=True, 
                                     monitor='val_loss', mode="min", verbose=1, save_best_only=True, period=1)
        
        tensorboard_callback = TensorBoard(LOG_FOLDER=LOG_FOLDER, histogram_freq=5)
        
        model.fit_generator(
            training_data_gen,
            epochs = epochs,
            validation_data=validation_data_gen,
            callbacks=[csv_logger, checkpoint, tensorboard_callback]
        )
    
    return model


def predict(model, image):
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    
    image_np = np.asarray(resized_image, dtype=np.float32)
    image_np = image_np[:, :, 0:3]
    image_np = imagenet_utils.preprocess_input(image_np, mode="tf")
    image_np = np.expand_dims(image_np, axis=0)
    
    result = model.predict(image_np)

    predicted_body_parts = np.argmax(result[0][:, :, 0:keypoint_index], axis=2)

    detected_keypoints = {}

    for i in range(keypoint_index, NUMBER_OF_KEYPOINTS + NUMBER_OF_BODY_PARTS + 1):
        squeezed_mask = result[0][:, :, i].squeeze()
    
        if (squeezed_mask.max() < 0.01):
            continue
        
        detected_keypoint = fi.gaussian_filter(squeezed_mask, SIGMA, truncate=11)
        max_index = detected_keypoint.argmax()
        y,x = np.unravel_index(max_index, detected_keypoint.shape)
        
        x *= (MASK_DOWNSAMPLING_FACTOR)
        y *= (MASK_DOWNSAMPLING_FACTOR)
        
        detected_keypoints[i - keypoint_index] = [x, y]

    return detected_keypoints, predicted_body_parts, resized_image


def predict_single_image(model, image_path):
    image_file_name = os.path.basename(image_path)  
    print(image_file_name)
    
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
    
    
if __name__ == '__main__':
    model = create_res_u_net_model()
    model.load_weights(os.path.join(LOG_FOLDER, MODEL_NAME))
    
    # model = train(model)
    
    if (debug):
        image_folder = TRAIN_IMAGE_FOLDER
        image_names = os.listdir(TRAIN_IMAGE_FOLDER)[:10]
                
    else :
        image_folder = TEST_IMAGE_FOLDER
        image_names = os.listdir(TEST_IMAGE_FOLDER)
        
    for image_name in image_names:
        predict_single_image(model, os.path.join(image_folder, image_name))
