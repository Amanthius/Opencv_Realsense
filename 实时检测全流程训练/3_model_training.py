# -*- coding: utf-8 -*-
"""
综合项目第三步：训练YOLOv8水果检测模型 (自动类别检测版)

功能:
- [新] 自动检测并读取 'fruit_dataset/labels/classes.txt' 文件来获取类别列表。
- 自动创建YOLOv8训练所需的 `data.yaml` 配置文件。
- 加载预训练的YOLOv8n模型。
- 使用您自定义的数据集进行训练。

使用方法:
1. 确保已完成数据收集和标注，并已将数据集划分为 train/val 两个部分。
2. 确保 `labelImg` 生成的 `classes.txt` 文件位于 `fruit_dataset/labels/` 文件夹中。
3. 运行此脚本开始训练。
"""
import os
import yaml
from ultralytics import YOLO

# --- 1. 配置参数 ---

# 数据集根目录
DATASET_DIR = 'fruit_dataset'
# 训练轮次
EPOCHS = 30
# 预训练模型
MODEL_NAME = 'yolov8n.pt' 

def get_class_names():
    """
    自动从 'fruit_dataset/labels/classes.txt' 文件读取类别名称。
    """
    classes_file_path = 'fruit_dataset/labels/classes.txt'
    
    if os.path.exists(classes_file_path):
        print(f"检测到类别文件: {classes_file_path}，正在自动读取...")
        try:
            with open(classes_file_path, 'r', encoding='utf-8') as f:
                # 读取所有行，并去除每行末尾的换行符
                class_names = [line.strip() for line in f if line.strip()]
            
            if class_names:
                print(f"成功读取到 {len(class_names)} 个类别: {class_names}")
                return class_names
            else:
                print(f"警告: '{classes_file_path}' 文件为空。")
        except Exception as e:
            print(f"读取类别文件时出错: {e}")
    
    # --- 安全回退 ---
    # 如果文件不存在或读取失败，使用手动定义的列表
    print("未找到或无法读取 'classes.txt'。将使用脚本中定义的手动列表。")
    print("请确保 'CLASS_NAMES' 变量在脚本中已正确设置。")
    # [备用方案] 在此处定义您的手动类别列表
    fallback_class_names = ['li', 'jv'] # 例如: ['apple', 'banana']
    return fallback_class_names

def create_yaml_file(class_names):
    """自动创建 data.yaml 配置文件"""
    
    if not class_names:
        print("错误: 类别列表为空，无法创建 data.yaml。")
        return False
        
    # 检查数据集路径是否存在
    if not os.path.exists(DATASET_DIR):
        print(f"错误: 数据集文件夹 '{DATASET_DIR}' 不存在。")
        return False
        
    print("正在创建 data.yaml 配置文件...")
    
    try:
        dataset_abs_path = os.path.abspath(DATASET_DIR)
    except Exception as e:
        print(f"获取绝对路径时出错: {e}")
        return False

    # 创建YAML文件内容
    yaml_content = {
        'train': os.path.join(dataset_abs_path, 'images/train/'),
        'val': os.path.join(dataset_abs_path, 'images/val/'),
        'nc': len(class_names),
        'names': class_names
    }

    # 写入文件
    yaml_file_path = os.path.join(DATASET_DIR, 'data.yaml')
    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"data.yaml 已成功创建在: {yaml_file_path}")
        return yaml_file_path
    except Exception as e:
        print(f"写入 data.yaml 文件时出错: {e}")
        return False

def main():
    # --- 2. 自动获取类别并创建配置文件 ---
    class_names_list = get_class_names()
    yaml_path = create_yaml_file(class_names_list)
    
    if not yaml_path:
        print("配置文件创建失败，训练中止。")
        return
        
    # --- 3. 加载预训练模型 ---
    print(f"\n正在加载预训练模型: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # --- 4. 开始训练 ---
    print("="*50)
    print("即将开始模型训练...")
    print(f"  - 数据集配置: {yaml_path}")
    print(f"  - 训练轮次 (Epochs): {EPOCHS}")
    print(f"  - 图像尺寸 (Image Size): 640")
    print("="*50 + "\n")

    try:
        # 使用数据集训练模型
        results = model.train(data=yaml_path, epochs=EPOCHS, imgsz=640, device=0) # device=0 使用GPU

        print("\n" + "*"*50)
        print("训练完成！")
        print(f"最优模型已保存在最新的 '{results.save_dir}/weights/best.pt' 文件中。")
        print("*"*50)

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("请检查：")
        print("1. ultralytics, torch, torchvision 等库是否已正确安装。")
        print("2. 数据集路径和文件是否都正确无误。")
        print("3. 如果您有GPU，请确保CUDA和cuDNN已正确配置。")

if __name__ == '__main__':
    main()
