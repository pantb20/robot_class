import os
import cv2
import numpy as np

def level3_contour_detection(image_path):
    # 1. 读取原图
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败，请检查路径: {image_path}")
        return None
        
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 阈值分割：只把画面中极亮的部分（> 235）提取出来变成白色，其余变黑
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # 3. 查找这些极亮区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lamp_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉太小的噪点，长条灯通常面积较大
        if area > 500: 
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            
            lamp_detected = True
            # 在原图上把找到的发光长条灯框出来
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "ON Lamp", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    if lamp_detected:
        print(f"文件: {os.path.basename(image_path)} -> 检测结果：发现点亮的 LED 灯 (ON)")
    else:
        print(f"文件: {os.path.basename(image_path)} -> 检测结果：未发现发光灯具 (OFF)")
        
    # ================= 核心：添加可视化展示窗口 =================
    # 为了防止图片太大超出屏幕，可以先等比例缩放（可选）
    # 我们可以把原图和二值化图横向拼在一起，或者分立开显示
    
    # 因为 thresh 是单通道黑白图，为了跟彩色图 img 一起显示，可以转成 3 通道
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # 弹窗显示 1：原始输入图片
    cv2.imshow("1. Original Image ", img)
    
    # 弹窗显示 2：阈值二值化处理后的效果（观察计算机眼里只有黑和白）
    cv2.imshow("2. Threshold Binary ", thresh_bgr)
    
    print("提示：请在弹出的图片窗口中按键盘任意键（如空格或回车），以查看下一张图片...")
    
    # cv2.waitKey(0) 表示程序暂停在这里，等待用户按键盘任意键才继续执行下一张
    cv2.waitKey(0)
    
    # 关闭所有 OpenCV 弹窗，准备进入下一张
    cv2.destroyAllWindows()
    
    return img

if __name__ == '__main__':
    # 定义测试目录
    train_on_dir = "dataset_split/val/off"
    
    print(f"=== 开始使用第三阶段 轮廓查找法（带可视化）测试目录: {train_on_dir} ===")
    
    if not os.path.exists(train_on_dir):
        print(f"错误：路径 {train_on_dir} 不存在，请先运行数据集划分脚本！")
    else:
        file_list = os.listdir(train_on_dir)
        success_count = 0
        
        for file_name in file_list:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(train_on_dir, file_name)
                
                # 调用带可视化的函数
                level3_contour_detection(image_path)
                success_count += 1
                
        print(f"=== 遍历结束，共测试了 {success_count} 张图片 ===")