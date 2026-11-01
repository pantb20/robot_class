import os
import cv2
import numpy as np

def level3_contour_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败，请检查路径: {image_path}")
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 阈值分割：只把画面中极亮的部分（> 235）提取出来变成白色，其余变黑
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # 2. 查找这些极亮区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lamp_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉太小的噪点，长条灯通常面积较大
        if area > 500: 
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            
            lamp_detected = True
            # 把找到的发光长条灯框出来
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "ON Lamp", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    if lamp_detected:
        print(f"文件: {os.path.basename(image_path)} -> 检测结果：发现点亮的 LED 灯 (ON)")
    else:
        print(f"文件: {os.path.basename(image_path)} -> 检测结果：未发现发光灯具 (OFF)")
        
    return img

if __name__ == '__main__':
    # 定义训练集里 'on'（开灯）文件夹的路径
    train_on_dir = "dataset_split/val/on"
    
    print(f"=== 开始使用第三阶段 轮廓查找法测试目录: {train_on_dir} ===")
    
    if not os.path.exists(train_on_dir):
        print(f"错误：路径 {train_on_dir} 不存在，请先运行数据集划分脚本！")
    else:
        file_list = os.listdir(train_on_dir)
        success_count = 0
        
        for file_name in file_list:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(train_on_dir, file_name)
                # 调用第三阶段函数
                level3_contour_detection(image_path)
                success_count += 1
                
        print(f"=== 遍历结束，共测试了 {success_count} 张开灯图片 ===")