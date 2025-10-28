# -*- coding: utf-8 -*-
"""
综合项目第三步：训练YOLOv8水果检测模型

功能:
- 自动创建YOLOv8训练所需的 `data.yaml` 配置文件。
- 加载预训练的YOLOv8n模型。
- 使用您在上一步中准备好的数据集进行训练。
- 训练完成后，最优的模型权重将保存在 'runs/detect/trainX/weights/best.pt'。

使用方法:
1. 确保您已经完成了数据收集和标注，并已将数据集划分为 train/val 两个部分。
2. 确保 `fruit_dataset` 文件夹与此脚本位于同一目录下。
3. 根据您自己的类别，修改下面的 `CLASS_NAMES` 列表。
4. 运行此脚本开始训练。
"""
import os
import yaml
from ultralytics import YOLO

# --- 1. 配置参数 ---

# 数据集根目录
DATASET_DIR = 'fruit_dataset'
# 定义您的类别名称，顺序必须和标注时一致！
# 例如，如果标注时 apple 是第0类，banana 是第1类
CLASS_NAMES = ['li', 'jv'] 
# 训练轮次
EPOCHS = 30
# 预训练模型 (yolov8n.pt 是最小最快的模型, yolov8s.pt/yolov8m.pt 精度更高但更慢)
MODEL_NAME = 'yolov8n.pt' 

def create_yaml_file():
    """自动创建 data.yaml 配置文件"""
    
    # 检查数据集路径是否存在
    if not os.path.exists(DATASET_DIR):
        print(f"错误: 数据集文件夹 '{DATASET_DIR}' 不存在。")
        print("请确保已创建该文件夹，并完成了数据集的准备。")
        return False
        
    print("正在创建 data.yaml 配置文件...")
    
    # 获取数据集的绝对路径
    try:
        dataset_abs_path = os.path.abspath(DATASET_DIR)
    except Exception as e:
        print(f"获取绝对路径时出错: {e}")
        return False

    # 创建YAML文件内容
    yaml_content = {
        'train': os.path.join(dataset_abs_path, 'images/train/'),
        'val': os.path.join(dataset_abs_path, 'images/val/'),
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    # 写入文件
    yaml_file_path = os.path.join(DATASET_DIR, 'data.yaml')
    try:
        with open(yaml_file_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print(f"data.yaml 已成功创建在: {yaml_file_path}")
        return yaml_file_path
    except Exception as e:
        print(f"写入 data.yaml 文件时出错: {e}")
        return False

def main():
    # --- 2. 创建配置文件 ---
    yaml_path = create_yaml_file()
    if not yaml_path:
        return
        
    # --- 3. 加载预训练模型 ---
    print(f"\n正在加载预训练模型: {MODEL_NAME}")
    # 加载一个预训练的YOLOv8模型
    model = YOLO(MODEL_NAME)

    # --- 4. 开始训练 ---
    print("="*50)
    print("即将开始模型训练...")
    print(f"  - 数据集配置: {yaml_path}")
    print(f"  - 训练轮次 (Epochs): {EPOCHS}")
    print(f"  - 图像尺寸 (Image Size): 640")
    print("="*50 + "\n")

    try:
        # 使用数据集训练模型
        # data: 指向你的 data.yaml 文件
        # epochs: 训练轮次
        # imgsz: 输入图像的尺寸
        results = model.train(data=yaml_path, epochs=EPOCHS, imgsz=640)

        print("\n" + "*"*50)
        print("训练完成！")
        print("最优模型已保存在最新的 'runs/detect/trainX/weights/best.pt' 文件中。")
        print("请将该 'best.pt' 文件复制到与第4步脚本相同的目录下，准备进行实时检测。")
        print("*"*50)

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("请检查：")
        print("1. ultralytics, torch, torchvision 等库是否已正确安装。")
        print("2. 数据集路径和文件是否都正确无误。")
        print("3. 如果您有GPU，请确保CUDA和cuDNN已正确配置。")

if __name__ == '__main__':
    main()
