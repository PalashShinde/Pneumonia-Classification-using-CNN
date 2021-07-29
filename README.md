# Pneumonia-Classification-using-CNN

This Flask application mainly helps in Detetcting Pneumonia for a given patient Chest X-Ray sample.

Application in NutShell :-
* User Uploads the image of Chest X-Ray sample.
* Application then calls Vgg16 CNN model to detect wheather a X-ray Sample has Pneumonia infection or not and returns back the sample with predicted class. 

![APP Demo](https://github.com/PalashShinde/Pneumonia-Classification-using-CNN/blob/main/app/app_gif/pnenomia_gg.gif)

### SetUP and Instruction Guide

#### Install Python Dependencies 
The `requirements.txt` file should list all Python libraries that your notebooks
depend on, and they will be installed using:
```
cd ~/app/
pip install -r requirements.txt
```
### Dataset
Model is trained on [kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of Chest X-Ray Images (Pneumonia) challenge
#### [Pretrained Vgg16 model weights](https://keras.io/api/applications/vgg/#vgg16-function)

### Model Training 
Place the train & val data from kaggle repo in below dirs~
```
cd ~/app/model_train/data/
$ treee
.
├── train_data
│   ├── train_n
│   │   ├── 0ab731ef-0580-4a2e-9fa2-29a243b71c86.png
│   │   ├── 0af57062-e071-4083-ae38-a6681b101994.png
│   └── train_p
│       ├── 0a0f91dc-6015-4342-b809-d19610854a21.png
│       ├── 0a2c130c-c536-4651-836d-95d07e9a89cf.png
│       ├── 0a2f6cf6-1f45-44c8-bcf0-98a3b466b597.png
│       └── 0a6a5956-58cf-4f17-9e39-7e0d17310f67.png
└── val_data
    ├── val_n
    │   ├── 0ac2c2ef-efb4-485c-b63a-69f19f32b704.png
    │   ├── 0b18dcfc-c526-435b-aea8-d8038aa224ef.png
    └── val_p
        ├── 0a03fbf6-3c9a-4e2e-89ce-c7629ae43a27.png
        ├── 0a8d486f-1aa6-4fcf-b7be-4bf04fc8628b.png
```
Update the constants paths in constant.py 
```
cd ~/app/scripts/
nano constants.py
train_image_dir =  '~'
val_image_dir = '~'
train_weights_path = '~'
test_weights_path = '~'
checkpoint_path = '~'
tensorboard_dir = '~'
```
Call tran.py script, Post training weights file is saved on training_weights path.
```
cd ~/app/model_train/scripts/
$ python3 classifier.py train
```

### Test weights with Flask application locally.
Update the weight file path in app.py on weight_path = STATIC_FOLDER + '/' + '.h5'
```
cd ~/app/
$ python3 app.py
```

