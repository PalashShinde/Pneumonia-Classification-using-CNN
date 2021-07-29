# Pneumonia-Classification-using-CNN

This Flask application mainly helps in Detetcting Pneumonia for a given patient Chest X-Ray sample.
Application in NutShell :-
1) user Uploads the image of Chest X-Ray sample.
2) Application then calls Vgg16 CNN model to detect wheather a X-ray Sample as Pneumonia infection or not. 

[APP Demo] ()

### SetUP and Instruction Guide

#### Install Python Dependencies 
The `requirements.txt` file should list all Python libraries that your notebooks
depend on, and they will be installed using:
```
cd ~/app/
pip install -r requirements.txt
```

### Vgg16 Weights 
#### Application requires FastPose & Yolov3 Model Weights Please refer [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) for more info.
##### To download [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights) & [FastPose](https://drive.google.com/u/0/uc?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn&export=download) Once completed, place the weights in below dirs ~
``` bash
~/app/detector/yolo/data/yolov3-spp.weights
```
``` bash
~ /app/pretrained_models/fast_res50_256x192.pth
```
#### Run Flask Application Locally.
``` bash
cd ~/app/
python3 app.py
```
