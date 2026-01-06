
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = '../data/models/pose_landmarker_full.task'

VIDEO_INPUT_PATH = "../data/3196221-uhd_3840_2160_25fps.mp4"
VIDEO_OUTPUT_PATH = "../data/pose_output2.mp4"
CSV_OUTPUT_PATH = "../data/pose_metrics2.csv"
TARGET_FPS = 24

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=5,
    min_pose_detection_confidence=0.7,
    # min_tracking_confidence=0.8,
    # min_pose_presence_confidence=0.8,
    output_segmentation_masks=False,
)

def draw_landmarks(frame, pose_landmarks):
    h, w, _ = frame.shape

    POSE_CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28)
    ]

    for lm in pose_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    for a, b in POSE_CONNECTIONS:
        x1, y1 = int(pose_landmarks[a].x * w), int(pose_landmarks[a].y * h)
        x2, y2 = int(pose_landmarks[b].x * w), int(pose_landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

orig_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, round(orig_fps / TARGET_FPS))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, TARGET_FPS, (w, h))

pose_records = []
frame_idx = 0
saved_frame_idx = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                for person_id, pose_landmarks in enumerate(result.pose_landmarks):
                    annotated = draw_landmarks(frame.copy(), pose_landmarks)

                    record = {
                        "frame": saved_frame_idx,
                        "person_id": person_id
                    }

                    for i, lm in enumerate(pose_landmarks):
                        record[f"lm_{i}_x"] = lm.x
                        record[f"lm_{i}_y"] = lm.y
                        record[f"lm_{i}_z"] = lm.z
                        record[f"lm_{i}_visibility"] = lm.visibility

                    pose_records.append(record)

            out.write(annotated if result.pose_landmarks else frame)

            saved_frame_idx += 1
            frame_idx += 1
            pbar.update(1)

cap.release()
out.release()

df = pd.DataFrame(pose_records)
df.to_csv(CSV_OUTPUT_PATH, index=False)

print("Done.")
print("Annotated video:", VIDEO_OUTPUT_PATH)
print("CSV saved:", CSV_OUTPUT_PATH)
print("Total rows:", len(df))


# with PoseLandmarker.create_from_options(options) as landmarker:
#     cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {VIDEO_INPUT_PATH}")
#         exit()
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
#     pose_data = []
    
#     frame_count = 0
#     with tqdm(total=total_frames, desc="Processing") as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # Calculate timestamp in milliseconds
#             frame_timestamp_ms = int(frame_count * 1000 / fps)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#             pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
#             annotated_image = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)
#             annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
#             out.write(annotated_bgr)
#             if pose_landmarker_result.pose_landmarks:
#                 for person_idx, pose_landmarks in enumerate(pose_landmarker_result.pose_landmarks):
#                     frame_data = {
#                         'frame': frame_count,
#                         'timestamp_ms': frame_timestamp_ms,
#                         'person_id': person_idx
#                     }
#                     for landmark_idx, landmark in enumerate(pose_landmarks):
#                         frame_data[f'landmark_{landmark_idx}_x'] = landmark.x
#                         frame_data[f'landmark_{landmark_idx}_y'] = landmark.y
#                         frame_data[f'landmark_{landmark_idx}_z'] = landmark.z
#                         frame_data[f'landmark_{landmark_idx}_visibility'] = landmark.visibility
                    
#                     pose_data.append(frame_data)
            
#             frame_count += 1
#             pbar.update(1)
    
#     # Release video resources
#     cap.release()
#     out.release()
    
#     # Save pose data to CSV
#     if pose_data:
#         df = pd.DataFrame(pose_data)
#         df.to_csv(CSV_OUTPUT_PATH, index=False)
#         print(f"\nPose metrics saved to: {CSV_OUTPUT_PATH}")
#         print(f"Total records: {len(df)}")
#     else:
#         print("\nNo pose landmarks detected in the video.")
    
#     print(f"Annotated video saved to: {VIDEO_OUTPUT_PATH}")
#     print("Processing complete!")






