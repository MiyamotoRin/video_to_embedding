import cv2
import os
import numpy as np
import tensorflow as tf

def video_embeddings(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the frames of the video
    frame_count = 0
    frames = []
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Check if we have reached the end of the video
        if not ret:
            break

        # Append a frame file every 10 frames
        if frame_count % 10 == 0:
            frames.append(frame)
        # Increment the frame count
        frame_count += 1

    # Release the video file
    cap.release()
    
    # ResNet50モデルを読み込む（最後の全結合層を含まない）
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # フレームのリストをnumpy配列に変換
    frames_np = np.array(frames)

    # 画像の前処理（ResNet50用の前処理を行う）
    preprocessed_frames = tf.keras.applications.resnet50.preprocess_input(frames_np)

    # 特徴ベクトルの抽出
    features = base_model.predict(preprocessed_frames)
    np.savetxt('features.csv', features, delimiter=',')
    

if __name__ == "__main__":
    video_path = "video.mp4"
    output_folder = "frames"
    video_embeddings(video_path, output_folder)