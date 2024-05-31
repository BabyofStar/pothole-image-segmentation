#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from ultralytics import YOLO
import cv2
from PIL import Image
import yaml
import random
import torch

#%%
dataDir = 'C:/Users/LX/Desktop/opencv/opencv-2/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8'

trainImagePath = os.path.join(dataDir, 'train','images')

#list of the images
imageFiles = [f for f in os.listdir(trainImagePath) if f.endswith('.jpg')]

randomImages = random.sample(imageFiles, 15)

plt.figure(figsize=(10, 10))

for i, image_file in enumerate(randomImages):

    image_path = os.path.join(trainImagePath, image_file)
    image = Image.open(image_path)
    plt.subplot(3, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')

  # Add a suptitle
plt.suptitle('Random Selection of Dataset Images', fontsize=24)

# Show the plot
plt.tight_layout()
plt.show()

model = YOLO('yolov8n-seg.pt')

yamlFilePath = os.path.join(dataDir,'data.yaml' )

results = model.train( 
    data= yamlFilePath, 
    epochs= 30 , 
    imgsz= 640 , 
    batch= 32 , 
    optimizer= 'auto' , 
    lr0= 0.0001 ,               # 初始学习率
    lrf= 0.01 ,                 # 最终学习率 (lr0 * lrf)
    dropout = 0.25 ,           # 使用 dropout 正则化
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu"), # 运行的设备
 )

#%%
bestModelpath = "C:/Users/LX/Desktop/opencv/opencv-2/runs/segment/train2/weights/best.pt"
bestModel = YOLO(bestModelpath)

validImagePath = os.path.join(dataDir, 'valid', 'images')

imageFiles = [f for f in os.listdir(validImagePath) if f.endswith('.jpg')]

#select Random images
numImages = len(imageFiles)
selectedImage = [imageFiles[i] for i in range(0, numImages, numImages // 9)]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Validation Set Inferences')

for i, ax in enumerate(axes.flatten()):
  imagePath = os.path.join(validImagePath, selectedImage[i])
  results = bestModel.predict(source= imagePath, imgsz=640)
  annotatedImage = results[0].plot()
  annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)
  ax.imshow(annotatedImageRGB)
  ax.axis('off')

plt.tight_layout()
plt.show()

#%%
import shutil
videoPath = "C:/Users/LX/Desktop/opencv/opencv-2/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8/sample_video.mp4"

bestModel.predict(source=videoPath, save=True)

#%%
import subprocess

# Convert AVI to MP4 using FFmpeg
subprocess.call(['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/content/runs/segment/predict/sample_video.avi', 'output_video.mp4'])

from IPython.display import Video

# Display the converted MP4 video
Video("output_video.mp4", embed=True, width=960)
# %%
