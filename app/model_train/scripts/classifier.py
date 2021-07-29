import utils
import constants
import sys
import cv2
import numpy as np
import os
from keras.utils import multi_gpu_model
import vgg16
import vgg_19
import csv
training_weights = constants.train_weights_path
test_weights = constants.test_weights_path
import pandas as pd

def train():
    
    model = vgg16.VGG16_model(weights=training_weights,img_shape = (256,256,1) , input_tensor=None,classes=2, include_top=True)
    model.compile(loss = 'binary_crossentropy',
                optimizer = constants.sgd,
                metrics = ['acc'])
    print('model.summary()', model.summary())
    print('len(model.layers)',len(model.layers))
    model.fit_generator(generator=utils.train_image_generator,
                    steps_per_epoch = constants.steps_per_epoch,
                    epochs = constants.epochs,
                    validation_data=utils.validation_image_generator, validation_steps= constants.steps_per_epoch ,
                    use_multiprocessing=False,
                    callbacks = [constants.checkpointer, constants.tensorboard, constants.LR_Reducer],
                    # workers = 2
                    )
    
    model.save_weights(training_weights)

if __name__ == '__main__':
    
    if sys.argv[1] == 'train':
        train()