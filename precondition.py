
#%%
import cv2
import os

# 定义源文件夹路径
source_folder_path = 'C:/Users/LX/Desktop/opencv/opencv-2/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8/train/images'

# 定义目标文件夹路径
target_folder_path = 'C:/Users/LX/Desktop/opencv/opencv-2/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8/train/pre_images'

# 创建目标文件夹，如果不存在的话
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 获取源文件夹中的所有图片文件
image_files = [f for f in os.listdir(source_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 对每个图片文件进行高斯滤波处理
for image_file in image_files:
    # 构建源文件的完整路径
    source_image_path = os.path.join(source_folder_path, image_file)
    # 构建目标文件的完整路径
    target_image_path = os.path.join(target_folder_path, image_file)
    
    # 读取图片
    image = cv2.imread(source_image_path)
    
    # 进行高斯滤波处理
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

    # 像素归一化处理
    normalized_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 保存处理后的图片
    cv2.imwrite(target_image_path, filtered_image)
    
    print(f'Processed and saved {image_file}')

print('All images have been processed and saved.')
# %%
