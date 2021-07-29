from keras_preprocessing.image import ImageDataGenerator
import constants
import keras.backend as K
from keras.layers import Flatten
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy
import cv2
from keras.utils import np_utils
import time
import random

train_image_dir = constants.train_image_dir
val_image_dir = constants.val_image_dir

img_width, img_height = constants.img_width, constants.img_height

batch_size = constants.batch_size
CLASSES = constants.CLASSES

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        #ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=45, p=.2),
        #OneOf([
        #    IAAAdditiveGaussianNoise(),
        #    GaussNoise(),
        #], p=0.9),
        #OneOf([
        #    MotionBlur(p=.2),
        #    MedianBlur(blur_limit=3, p=.1),
        #    Blur(blur_limit=3, p=.1),
        #], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2)
        #OneOf([
            # CLAHE(clip_limit=2),
        #    IAASharpen(),
        #    IAAEmboss(),
            # RandomContrast(),
            # RandomBrightness(),
        #], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)

def classify_generator(folder_path):
    augmentation = strong_aug(p=0.9)
    class_dict = dict(enumerate(os.listdir(folder_path)))
    # class_dict {0: 'train_p', 1: 'train_n'}
    mappings = []
    for fol in class_dict.keys():
        for i in os.listdir(folder_path+class_dict[fol]):
            mappings.append((folder_path+class_dict[fol]+'/'+i, fol))

    for _ in range(constants.epochs * constants.steps_per_epoch//constants.batch_size):

        x_train, y_train = [], []

        for s in random.sample(mappings, constants.batch_size):
            
            resized = cv2.resize( cv2.imread(s[0])[:,:,0] , (256,256)) 

            data = {"image": np.expand_dims(resized, -1)}

            augmented = augmentation(**data)

            x_train.append(augmented["image"])
            y_train.append([1.,0.] if not s[1] else [0.,1.])
            # y_train.append(s[1])

        x, y =  np.array(x_train), np.array(y_train)
        yield x, y

def myGenerator(train_generator):

    while True:
        xy = train_generator.next()
        yield xy


def image_data_generator():
    image_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        # fill_mode='nearest',
        validation_split=0.1)
    return image_data_generator

def image_generator():
    idg = image_data_generator()
    train_image_generator = idg.flow_from_directory(
        train_image_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        # class_mode='binary',
        class_mode='binary',
        color_mode="grayscale",
        # color_mode="rgb",
        shuffle=False,
        subset='training')

    validation_image_generator = idg.flow_from_directory(
        val_image_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        # class_mode='binary',
	    class_mode='binary',
        color_mode="grayscale",
        shuffle=False,
        subset='validation')

    return train_image_generator, validation_image_generator

# def img_512_generator(image_files, batch_size=1):
#     masks = [x.replace("tiles", "masks") for x in image_files]
#     number_of_imgs = len(image_files)
#     number_of_masks = len(masks)
#     print("len of images and masks {0}, {1}".format(
#         str(number_of_imgs), str(number_of_masks)))
#     assert number_of_imgs == number_of_masks
#     assert number_of_imgs > 0
#     augmentation = utils.strong_aug(p=0.9)

#     while True:
#         # Select files (paths/indices) for the batch
#         batch_indices = np.random.choice(a=number_of_imgs,
#                                          size=batch_size)
#         batch_input = []
#         batch_output = []

#         # Read in each input, perform preprocessing and get labels
#         for input_index in batch_indices:
#             # print("reading image from {0} and corr mask from {1}".format(image_files[input_index], masks[input_index]))
#             image = cv2.resize(cv2.imread(image_files[input_index]), (1024, 1024))
#             mask = cv2.resize(cv2.imread(masks[input_index]), (1024, 1024))

#             data = {"image": image, "mask": mask}

#             augmented = augmentation(**data)

#             image, mask = augmented["image"], augmented["mask"]

#             output = utils.getSegmentationArr(mask)
#             # output = utils.getBGSegmentationArr(mask)


#             batch_input += [image[:512,:512], image[512:,:512], image[:512,512:], image[512:,:512:]]
#             batch_output += [output[:512,:512], output[512:,:512], output[:512,512:], output[512:,:512:]]
#             #images = [image[:512,:512], image[512:,:512], image[:512,512:], image[512:,:512:]]
#             #outputs = [output[:512,:512], output[512:,:512], output[:512,512:], output[512:,:512:]]
#             for i in range(len(images)):
#                  batch_input += images[i]
#                  batch_output += outputs[i]

#             # # batch_input += [image]
#             # # batch_output += [output]

#             batch_x = np.array([images[i]])
#             # YOU CAN SPECIFY THE OUTPUT in BATCH_Y
#             batch_y = np.array([outputs[i]])

#             print(batch_x.shape, batch_y.shape)

#             yield (batch_x, batch_y)



# def multi_to_binary(rgb_img, mask_img, str_class):
#     batch_input = []
#     batch_output = []
#     image = rgb_img
#     # image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
#     mask = mask_img
#     images = [image[:256, :256], image[256:, :256],
#               image[:256, 256:], image[256:, :256:]]
#     outputs = [mask[:256, :256], mask[256:, :256],
#                mask[:256, 256:], mask[256:, :256:]]

#     for img, mask in zip(images, outputs):
#         if set([0, 255]).issubset(np.unique(mask).tolist()):
#             batch_input += [img]
#             batch_output += [int(str_class)]
#         else:
#             batch_input += [img]
#             batch_output += [0]

#     return batch_input, batch_output

# def multi_to_binary(img_name,image_files_path,str_class):
    ################# current one 
#     batch_input = []
#     batch_output = []
#     image = cv2.resize(cv2.imread(os.path.join(image_files_path+'train/')+img_name), (512, 512))
#     image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
#     mask = cv2.resize(cv2.imread(os.path.join(image_files_path+'masks/')+img_name), (512, 512))
#     images = [image[:256,:256], image[256:,:256], image[:256,256:], image[256:,:256:]]
#     outputs = [mask[:256,:256], mask[256:,:256], mask[:256,256:], mask[256:,:256:]]

#     # for i in range(len(images)):
#     #     # print(len(images))
#     #     if set([0,255]).issubset(np.unique(outputs[i]).tolist()):
#     #         batch_input += [images[i]]
#     #         batch_output+= [int(str_class)]
#     #     else:
#     #         batch_input += [images[i]]
#     #         batch_output+= [0]

#     for img,mask in zip(images,outputs):
#         if set([0,255]).issubset(np.unique(mask).tolist()):
#             batch_input += [img]
#             batch_output+= [int(str_class)]
#         else:
#             batch_input += [img]
#             batch_output+= [0]

#     return batch_input, batch_output


# def img_512_generator(image_files_path):

#     t_path = os.path.join(image_files_path+'train/')
#     m_path = os.path.join(image_files_path+'masks/')

#     train_files = [t_path+x for x in os.listdir(t_path) if x.endswith(".png")]
#     mask_files = [t_path.replace(
#         "train", "masks")+x for x in os.listdir(t_path) if x.endswith(".png")]

#     print('Start ', time.time())
#     # train_img_list = [cv2.imread(x) for x in train_files]
#     # mask_img_list = [cv2.imread(x) for x in mask_files]
#     print('End ', time.time())

#     # number_of_imgs = len(train_img_list)
#     # number_of_masks = len(mask_img_list)
#     # print("len of images and masks {0}, {1}".format(str(number_of_imgs), str(number_of_masks)))
#     # assert number_of_imgs == number_of_masks
#     # assert number_of_imgs > 0
#     loop_idx = 0

#     while loop_idx < len(train_files) * constants.epochs:
#         idx = loop_idx%len(train_files)
#         # print("index is", str(idx))
#         class_id = train_files[idx].split("/")[-1].split("-")[0]
#         img = cv2.imread(train_files[idx])
#         mask = cv2.imread(mask_files[idx])


#         # batch_input_final = []
#         # batch_output_final = []

#         # batch_input, batch_output = multi_to_binary(train_img_list[idx], mask_img_list[idx], class_id)
#         batch_input, batch_output = multi_to_binary(img, mask, class_id)


#         # if class_id == '1':# == train_img_list[idx][2].split('-')[0]:
#         #     batch_input, batch_output = multi_to_binary(train_img_list[idx],mask_img_list[idx],class_id)
#         # elif class_id == '2':
#         #     batch_input, batch_output = multi_to_binary(train_img_list[idx][0],mask_img_list[idx],class_id)
#         # elif class_id == '3':# '3' == train_img_list[idx][2].split('-')[0]:
#         #     batch_input, batch_output = multi_to_binary(train_img_list[idx][0],mask_img_list[idx],class_id)
#         # elif class_id == '4':# == train_img_list[idx][2].split('-')[0]:
#         #     batch_input, batch_output = multi_to_binary(train_img_list[idx][0],mask_img_list[idx],class_id)

#         # batch_input_final = batch_input
#         # batch_output_final = batch_output

#         batch_x = np.array(batch_input)
#         # YOU CAN SPECIFY THE OUTPUT in BATCH_Y
#         batch_y = np.array(batch_output)

#         batch_y_c = np_utils.to_categorical(batch_y, num_classes=5)
#         # loop_idx += 1
#         yield (batch_x, batch_y_c)


# def img_512_generator(image_files_path):
#     ################# current one 
#     train_img_list = sorted(os.listdir(os.path.join(image_files_path+'train/')))
#     mask_img_list = sorted(os.listdir(os.path.join(image_files_path+'masks/')))

#     # masks = [ x.replace("tiles", "masks") for x in image_files]
#     number_of_imgs = len(train_img_list)
#     number_of_masks = len(mask_img_list)
#     print("len of images and masks {0}, {1}".format(
#         str(number_of_imgs), str(number_of_masks)))
#     assert number_of_imgs == number_of_masks
#     assert number_of_imgs > 0
#     # augmentation = utils.strong_aug(p=0.9)

#     while True:
#         # Select files (paths/indices) for the batch
#         # batch_indices = np.random.choice(a=number_of_imgs,
#         #                                  size=batch_size)
#         # Read in each input, perform preprocessing and get labels
#         for img_name in train_img_list:

#             batch_input_final = []
#             batch_output_final = []

#             if '1' == img_name.split('-')[0]:
#                 str_class = img_name.split('-')[0]
#                 batch_input, batch_output = multi_to_binary(img_name,image_files_path,str_class)
#                 # batch_input_final.extend(batch_input)
#                 # batch_output_final.extend(batch_output)
#                 # batch_input_final += batch_input
#                 # batch_output_final+= batch_output
#             elif '2' == img_name.split('-')[0]:
#                 str_class = img_name.split('-')[0]
#                 batch_input, batch_output = multi_to_binary(img_name,image_files_path,str_class)
#                 # batch_input_final += batch_input
#                 # batch_output_final+= batch_output
#             elif '3' == img_name.split('-')[0]:
#                 str_class = img_name.split('-')[0]
#                 batch_input, batch_output = multi_to_binary(img_name,image_files_path,str_class)
#                 # batch_input_final += batch_input
#                 # batch_output_final+= batch_output
#             elif '4' == img_name.split('-')[0]:
#                 str_class = img_name.split('-')[0]
#                 batch_input, batch_output = multi_to_binary(img_name,image_files_path,str_class)

#             batch_input_final  += batch_input
#             batch_output_final += batch_output

#             batch_x = np.array(batch_input_final)
#             # YOU CAN SPECIFY THE OUTPUT in BATCH_Y
#             batch_y = np.array(batch_output_final)

#             batch_y_c = np_utils.to_categorical(batch_y,num_classes=5)

#             # print('shapessss2222',batch_x.shape, batch_y_c.shape)

#             yield (batch_x, batch_y_c)

# train_image_generator, validation_image_generator = image_generator()
# train_generator = myGenerator(train_image_generator)

train_image_generator = classify_generator(train_image_dir)
validation_image_generator = classify_generator(val_image_dir)

# train_image_generator = img_512_generator(train_image_dir)
# validation_generator = myGenerator(validation_image_generator)
