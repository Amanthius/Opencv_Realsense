# -*- coding: utf-8 -*-
"""
综合项目第一步：水果数据收集 


"""
import cv2
import pyrealsense2 as rs
import numpy as np
import os

# --- 配置参数 ---
SAVE_PATH = "fruit_dataset/images"
IMG_WIDTH = 640
IMG_HEIGHT = 480
FPS = 30
WINDOW_NAME = 'Realsense - 数据收集'

def main():
    # --- 1. 准备工作 ---
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # --- 2. 预先创建并初始化窗口 ---
    print("正在初始化GUI窗口...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("窗口初始化成功。")

    # --- 3. 初始化并启动 Realsense ---
    pipeline = None
    try:
        print("初始化Realsense Pipeline...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)

        print("正在启动摄像头...")
        pipeline.start(config)
        print("摄像头已启动！")

        # --- 4. 主循环 ---
        img_counter = 0
        print("\n" + "="*50)
        print("操作指南:")
        print("  - 窗口现在应该已显示，请将水果置于镜头前。")
        print("  - 按 's' 键保存当前画面。")
        print("  - 按 'q' 键 或 点击窗口的'X'按钮 退出程序。")
        print("="*50 + "\n")

        # 只要窗口可见，就持续循环
        while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if color_image.size == 0:
                continue

            # 在预先创建的窗口中显示图像
            cv2.imshow(WINDOW_NAME, color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("接收到 'q' 键，退出程序。")
                break
            elif key == ord('s'):
                img_name = os.path.join(SAVE_PATH, f"fruit_{img_counter:04d}.jpg")
                cv2.imwrite(img_name, color_image)
                print(f"已保存: {img_name}")
                img_counter += 1
        
        print("窗口已关闭。")

    except Exception as e:
        print(f"\n[!!] 程序发生严重错误: {e}")
        print("如果问题仍然存在，请尝试更新您的显卡驱动和Intel RealSense SDK版本。")

    finally:
        # --- 5. 清理资源 ---
        if pipeline:
            print("正在关闭摄像头...")
            pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已完全关闭。")

if __name__ == '__main__':
    main()
