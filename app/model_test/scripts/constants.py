img_width, img_height = 512, 512
CLASSES = 2
batch_size = 1
steps_per_epoch = 7851 // batch_size

train_image_dir =  '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/train/'
val_image_dir = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/test_val/val/'

train_weights_path = '/home/kalpit/User/palash_konverge/kaggle-data/resnet101_imagenet_1000_no_top.h5'
# train_weights_path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/best_weights/Feb25-vgg16_weights.39-0.4051-softmax.h5'
test_weights_path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/best_weights/Feb22-vgg16_weights.39-8.2269.h5'

checkpoint_path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/best_weights/Feb29-vgg16_weights.{epoch:02d}-{loss:.4f}-softmax.h5'
tensorboard_dir = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/resnet_34_new/logs/'

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adadelta , Adam
# from keras.optimizers import Adam, SGD, RMSprop
LR_Reducer = ReduceLROnPlateau(patience=10, monitor='loss', factor=0.95, verbose=1)
checkpointer = ModelCheckpoint(monitor='loss',
                               filepath=checkpoint_path, verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

adadelta = Adadelta(lr=0.01, rho=0.00001)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.85)
adam = Adam(lr=0.0001, decay=1e-5)
epochs = 2000

