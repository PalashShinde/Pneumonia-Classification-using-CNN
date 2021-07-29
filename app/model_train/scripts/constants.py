import os
img_width, img_height = 512, 512
CLASSES = 2
batch_size = 1
steps_per_epoch = 7851 // batch_size
dir_path = os.path.dirname(os.path.realpath(__file__)) 

train_image_dir =  '~'
val_image_dir = '~'

train_weights_path = '~'
test_weights_path = '~'

checkpoint_path = '~'
tensorboard_dir = '~'

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adadelta , Adam
LR_Reducer = ReduceLROnPlateau(patience=10, monitor='loss', factor=0.95, verbose=1)
checkpointer = ModelCheckpoint(monitor='loss',
                               filepath=checkpoint_path, verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

adadelta = Adadelta(lr=0.01, rho=0.00001)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.85)
adam = Adam(lr=0.0001, decay=1e-5)
epochs = 2000

