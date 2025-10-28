# -*- coding: utf-8 -*-
"""
综合项目第四步：模型部署与实时检测
功能:
- 启动Realsense D435摄像头。
- 加载您在第三步中训练好的自定义YOLOv8模型。
- 对实时视频流进行水果检测。
- 在画面上绘制检测框、类别和置信度。
- 按下 'q' 键退出程序。

[!!] 调试功能 [!!]
- 引入可调的 CONFIDENCE_THRESHOLD，用于控制检测灵敏度。
- 在终端实时打印每一帧的详细检测结果（类别和置信度），
  用于诊断模型是否工作以及其置信度水平。
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

# --- 1. 配置参数 ---
# 图像尺寸
IMG_WIDTH = 640
IMG_HEIGHT = 480
# 帧率
FPS = 30
# 置信度阈值。您可以降低此值来查看置信度较低的检测结果
CONFIDENCE_THRESHOLD = 0.2
# 请根据您实际的训练输出文件夹修改 'train' 部分 (例如 train2, train3...)
MODEL_PATH = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')

# 如果您只是想测试摄像头和YOLOv8，可以使用预训练模型
# MODEL_PATH = 'yolov8n.pt'  

def main():
    # --- 2. 加载自定义YOLOv8模型 ---
    print(f"正在加载自定义模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 在路径 '{MODEL_PATH}' 下未找到模型文件。")
        print("请检查: 1. 路径是否正确; 2. 模型是否已成功训练。")
        return
        
    try:
        model = YOLO(MODEL_PATH)
        print("模型加载成功！")
        # 打印模型识别的类别
        print(f"模型可识别的类别: {model.names}")
    except Exception as e:
        print(f"错误: 加载模型失败。{e}")
        return

    # --- 3. 配置并启动Realsense摄像头 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)

    print("正在启动Realsense摄像头...")
    try:
        pipeline.start(config)
        print("摄像头启动成功！")
    except Exception as e:
        print(f"错误：无法启动Realsense摄像头。{e}")
        return

    # --- 4. 主循环：实时检测 ---
    print("\n实时检测已启动，按 'q' 键退出。")
    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 将帧数据转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            # --- [!!] YOLOv8 推理，并传入置信度阈值 ---
            results = model(color_image, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # --- [!!] 调试代码，检查并打印检测结果 ---
            boxes = results[0].boxes
            # 打印当前帧找到的物体数量
            print(f"当前帧找到 {len(boxes)} 个物体。")
            # 如果找到了物体，打印它们的详细信息
            if len(boxes) > 0:
                for box in boxes:
                    # 获取类别ID和置信度
                    try:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        print(f"  -> 类别: {model.names[cls_id]}, 置信度: {confidence:.2f}")
                    except IndexError:
                        print("  -> 无法解析某个检测框的信息。")

            # --- 可视化结果 ---
            # results[0].plot() 会自动使用传入的conf阈值来绘制图像
            annotated_frame = results[0].plot()

            # 在窗口中显示带有检测结果的图像
            cv2.imshow("YOLOv8 Real-Time Detection (Debug Mode)", annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- 5. 停止摄像头并关闭窗口 ---
        print("正在关闭摄像头...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已成功关闭。")

if __name__ == '__main__':
    main()

