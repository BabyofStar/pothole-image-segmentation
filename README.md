# pothole-image-segmentation
本项目提出了一种基于YOLOv8的路面坑洞检测任务方法，实现了对坑洞的自动化识别和定位。

### 开发环境
windows 11  

python 3.11  

Ultralytics  

以及各种库（详见代码）  


### 数据集
来源于kaggle网站的[Pothole Image Segmentation Dataset](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset)  

请前往网站自行下载

### 功能预览
![效果图片](output.png)

### 运行代码前的一些准备工作
#### 设置路径数据
` dataDir = '/content/Pothole_Segmentation_YOLOv8/' `  

dataDir目录路径“/content/Pothole_Segmentation_YOLOv8/”的变量。该变量表示存储图像分割任务的数据集的目录。  
#### 加载性能最佳的模型
在模型训练完成后，需要加载性能最佳的模型  
` bestModelpath = '/content/runs/segment/train/weights/best.pt `  
`bestModelpath`该变量包含训练期间获得的最佳模型的文件路径。
