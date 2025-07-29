import cv2
import os

def extract_frames_from_mkv(video_path, output_dir, frames_per_second):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        raise ValueError("无法获取视频帧率")

    # 计算跳帧间隔
    frame_interval = int(video_fps / frames_per_second)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(output_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"提取完成，共保存了 {saved_count} 帧图像。")

if __name__ == "__main__":
    extract_frames_from_mkv("E:/OBS/2025-07-25_09-50-20.mkv", "output_frames", 30)

