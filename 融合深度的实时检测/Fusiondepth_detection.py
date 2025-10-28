# -*- coding: utf-8 -*-
"""
综合项目第二步：融合深度的目标检测

功能:
- 继承项目一的所有功能（加载YOLOv8模型，实时检测物体）。
- **功能**:
  - 对每一个检测到的物体，计算其边界框的中心点。
  - 利用Realsense的对齐深度帧，精确获取该中心点的深度距离。
  - 在画面上实时显示物体的类别、置信度以及三维空间距离。
  - 按下 's' 键可以保存一张带有完整标注（检测框+深度）的静态图像。

使用方法:
1. 确保此脚本与 `runs` 文件夹在同一级目录下。
2. 运行此脚本。
3. 将您训练过的物体放在摄像头前，观察实时检测和测距效果。
4. 按 's' 键保存截图，按 'q' 键退出。
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
import time

# --- 1. 配置参数 ---
# 图像尺寸
IMG_WIDTH = 640
IMG_HEIGHT = 480
# 帧率
FPS = 30
# 置信度阈值
CONFIDENCE_THRESHOLD = 0.5
# 模型路径 (请根据您实际的训练输出文件夹修改 'train' 部分)
MODEL_PATH = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
MODEL_PATH = 'yolov8n.pt'  # 如果没有自定义模型，可以使用预训练模型测试 
# 截图保存路径
SNAPSHOT_PATH = "snapshots"

class RealsenseCamera:
    """一个封装了Realsense摄像头所有操作的类，确保稳定运行并提供对齐帧。"""
    def __init__(self, width=IMG_WIDTH, height=IMG_HEIGHT, fps=FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.align = rs.align(rs.stream.color)

    def start(self):
        """启动摄像头并进行预热"""
        print("正在启动Realsense摄像头...")
        try:
            self.pipeline.start(self.config)
            print("摄像头启动成功！")
            # 等待自动曝光/增益稳定
            for _ in range(30):
                self.pipeline.wait_for_frames()
            return True
        except Exception as e:
            print(f"错误：无法启动Realsense摄像头。{e}")
            return False

    def get_aligned_frames(self):
        """获取对齐后的彩色帧和深度帧对象"""
        try:
            frames = self.pipeline.wait_for_frames(5000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
        except Exception as e:
            print(f"获取帧时发生错误: {e}")
            return None, None

    def stop(self):
        """停止摄像头"""
        print("正在关闭摄像头...")
        self.pipeline.stop()

def main():
    # --- 2. 加载模型与准备文件夹 ---
    print(f"正在加载自定义模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 在路径 '{MODEL_PATH}' 下未找到模型文件。")
        return
    try:
        model = YOLO(MODEL_PATH)
        print("模型加载成功！")
        print(f"模型可识别的类别: {model.names}")
    except Exception as e:
        print(f"错误: 加载模型失败。{e}")
        return

    if not os.path.exists(SNAPSHOT_PATH):
        os.makedirs(SNAPSHOT_PATH)

    # --- 3. 启动摄像头 ---
    cam = RealsenseCamera()
    if not cam.start():
        return

    # --- 4. 主循环：实时检测与测距 ---
    print("\n实时检测与测距已启动。")
    print("  - 按 's' 键保存截图。")
    print("  - 按 'q' 键退出。")
    try:
        while True:
            # 获取对齐的帧
            color_image, depth_frame = cam.get_aligned_frames()
            if color_image is None:
                continue

            # 使用YOLOv8进行推理
            results = model(color_image, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 遍历检测结果
            for box in results[0].boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 获取置信度和类别
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # --- [核心功能] 计算中心点并获取深度 ---
                # 计算边界框的中心点
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 从深度帧中获取该点的距离 (单位：米)
                # get_distance会自动处理深度比例因子
                distance = depth_frame.get_distance(cx, cy)
                
                # 如果距离有效 (大于0)
                if distance > 0:
                    # --- 可视化 ---
                    # 绘制边界框
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 绘制中心点
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    # 准备标签和距离文本
                    label = f"{class_name} {conf:.2f}"
                    distance_text = f"{distance:.2f}m"
                    # 显示类别标签
                    cv2.putText(color_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # 显示距离文本
                    cv2.putText(color_image, distance_text, (cx + 10, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            # 显示结果图像
            cv2.imshow("YOLOv8 Detection with Depth", color_image)
            key = cv2.waitKey(1) & 0xFF

            # 按 'q' 键退出
            if key == ord('q'):
                break
            # 按 's' 键保存截图
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                snapshot_file = os.path.join(SNAPSHOT_PATH, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snapshot_file, color_image)
                print(f"截图已保存至: {snapshot_file}")


    finally:
        # --- 5. 清理资源 ---
        cam.stop()
        cv2.destroyAllWindows()
        print("程序已成功关闭。")

if __name__ == '__main__':
    main()
