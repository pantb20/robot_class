import os
import cv2
import numpy as np

def level1_global_brightness(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败，请检查路径: {image_path}")
        return None
    
    # 2. 转换为灰度图（把彩色图变成单通道黑白图）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 计算全图平均亮度 (0-255)
    mean_brightness = np.mean(gray)
    
    # 4. 简单设定一个阈值判断
    if mean_brightness > 120:
        status = "ON (可能开灯)"
    else:
        status = "OFF (可能关灯)"
        
    print(f"图片: {os.path.basename(image_path)} | 平均亮度: {mean_brightness:.2f} | 检测状态: {status}")
    return status

if __name__ == '__main__':
    # 定义训练集里 'on'（开灯）文件夹的路径
    train_on_dir = "dataset_split/train/on"
    
    print(f"=== 开始遍历测试目录: {train_on_dir} ===")
    
    if not os.path.exists(train_on_dir):
        print(f"错误：路径 {train_on_dir} 不存在，请先运行数据集划分脚本！")
    else:
        # 获取文件夹下的所有文件
        file_list = os.listdir(train_on_dir)
        
        success_count = 0
        for file_name in file_list:
            # 过滤只处理图片文件
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(train_on_dir, file_name)
                # 调用第一阶段的函数进行检测
                level1_global_brightness(image_path)
                success_count += 1
                
        print(f"=== 遍历结束，共测试了 {success_count} 张开灯图片 ===")