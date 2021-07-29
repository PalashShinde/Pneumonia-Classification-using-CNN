import utils
import constants
import sys
import cv2
import numpy as np
import os
from keras.utils import multi_gpu_model
import vgg16
import vgg_19
import resnet_model
import csv
training_weights = constants.train_weights_path
test_weights = constants.test_weights_path
import pandas as pd

def train():

    # model  = resnet_model.load_ResNet18(img_shape = (256,256,1) , input_tensor=None, weights=training_weights, classes=2, include_top=True)
    # model  = resnet_model.load_resNet34(img_shape = (256,256,1), input_tensor=None, weights=training_weights, classes=2, include_top=True)
    # model  = resnet_model.load_resNet50(weights=training_weights, img_shape = (256,256,1) , input_tensor=None, classes=2, include_top=True)
    model  = resnet_model.load_resnet101(weights=training_weights,img_shape = (256,256,1) , input_tensor=None,classes=2, include_top=True)
    # model  = resnet_model.load_resnet152(weights=training_weights,img_shape = (256,256,1) , input_tensor=None,classes=2, include_top=True)
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

def test():
    
    model  = resnet_model.load_resnet152(weights=None,img_shape = None , input_tensor=None ,classes=2, include_top=True)
    print('model.summary()', model.summary())
    print(test_weights)
    print('start')
    path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/test/total_imgs/'
    for img_name in os.listdir(path):
        image = cv2.resize( cv2.imread(os.path.join(path,img_name))[:,:,0] , (128,128))
        image = np.expand_dims(image, 0)
        op = model.predict(image)
        predicted_list = op.tolist()[0]
        print('output: ', img_name, predicted_list)

if __name__ == '__main__':
    
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()