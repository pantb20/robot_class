import os
import cv2
import numpy as np

def level2_roi_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败，请检查路径: {image_path}")
        return None, None
        
    h, w, _ = img.shape
    
    # 假设我们通过观察，知道长条灯大约在画面的指定区域 (x, y, w, h)
    # 学生可以根据实际照片的灯具位置手动调整这几个比例
    rx, ry, rw, rh = int(w * 0.4), int(h * 0.1), int(w * 0.2), int(h * 0.4)
    
    # 截取灯具 ROI 区域
    roi = img[ry:ry+rh, rx:rx+rw]
    
    # 转换到 HSV 色彩空间，提取 V（明度）通道
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # 计算 ROI 区域的最高亮度及极高亮像素占比
    max_val = np.max(v_channel)
    # 避免分母为 0 的情况
    if (rw * rh) > 0:
        high_bright_ratio = np.sum(v_channel > 230) / (rw * rh)
    else:
        high_bright_ratio = 0
        
    # 判定：如果最高亮度极高，且高亮面积达到一定比例，说明开灯
    if max_val > 240 and high_bright_ratio > 0.2:
        status = "ON (灯已点亮)"
        color = (0, 255, 0) # 绿色
    else:
        status = "OFF (灯已熄灭)"
        color = (0, 0, 255) # 红色
        
    print(f"文件: {os.path.basename(image_path)} | ROI最高亮度: {max_val} | 高亮占比: {high_bright_ratio:.2f} | 状态: {status}")
    
    # 在原图标出检测框并展示结果
    cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), color, 3)
    cv2.putText(img, status, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img, status

if __name__ == '__main__':
    # 定义训练集里 'on'（开灯）文件夹的路径
    train_on_dir = "dataset_split/train/off"
    
    print(f"=== 开始使用第二阶段 ROI 区域法测试目录: {train_on_dir} ===")
    
    if not os.path.exists(train_on_dir):
        print(f"错误：路径 {train_on_dir} 不存在，请先运行数据集划分脚本！")
    else:
        file_list = os.listdir(train_on_dir)
        success_count = 0
        
        for file_name in file_list:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(train_on_dir, file_name)
                # 调用第二阶段函数
                level2_roi_detection(image_path)
                success_count += 1
                
        print(f"=== 遍历结束，共测试了 {success_count} 张开灯图片 ===")