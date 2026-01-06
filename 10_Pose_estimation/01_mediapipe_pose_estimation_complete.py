import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
model_path = '../data/models/pose_landmarker_full.task'
VIDEO_INPUT_PATH = "../data/input_video.mp4"
VIDEO_OUTPUT_PATH = "../data/pose_output.mp4"
CSV_OUTPUT_PATH = "../data/pose_metrics.csv"

# Setup MediaPipe Pose Landmarker using tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure options for video mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False,
)

# Drawing utilities - simple circle and line drawing
def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Manually draws pose landmarks and connections on the image.
    
    Logic:
    - MediaPipe pose has 33 landmarks
    - We draw circles for each landmark
    - We draw lines between connected landmarks (skeleton)
    """
    if not detection_result.pose_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape
    
    # Define pose connections (skeleton structure)
    # These are the standard MediaPipe pose connections
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    for pose_landmarks in detection_result.pose_landmarks:
        # Convert normalized coordinates to pixel coordinates
        landmark_points = []
        for landmark in pose_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_points.append((x, y))
        
        # Draw connections (lines between landmarks)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks (circles)
        for point in landmark_points:
            cv2.circle(annotated_image, point, 5, (0, 0, 255), -1)
    
    return annotated_image

# Main processing
print("Initializing pose landmarker...")

with PoseLandmarker.create_from_options(options) as landmarker:
    # Open the input video using OpenCV
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_INPUT_PATH}")
        exit()
    
    # Get video properties
    # Logic: We need FPS to calculate timestamps, and dimensions for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    # Data storage for CSV export
    pose_data = []
    
    # Frame counter for timestamp calculation
    frame_count = 0
    
    # Process video frame by frame
    print("Processing video frames...")
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while cap.isOpened():
            # Read frame from video
            ret, frame = cap.read()
            
            if not ret:
                break  # End of video
            
            # Calculate timestamp in milliseconds
            # Logic: Each frame occurs at (frame_number / fps) seconds
            # Multiply by 1000 to convert seconds to milliseconds
            frame_timestamp_ms = int(frame_count * 1000 / fps)
            
            # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
            # Logic: OpenCV uses BGR, but MediaPipe expects RGB color order
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image object
            # Logic: MediaPipe needs its own image format wrapper
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform pose detection on the frame
            # Logic: Pass the image and timestamp to get pose landmarks
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            # Draw landmarks on the frame
            annotated_image = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)
            
            # Convert back to BGR for OpenCV video writer
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Write annotated frame to output video
            out.write(annotated_bgr)
            
            # Extract landmark data for CSV
            # Logic: Store coordinates of all detected landmarks for later analysis
            if pose_landmarker_result.pose_landmarks:
                for person_idx, pose_landmarks in enumerate(pose_landmarker_result.pose_landmarks):
                    # Create a record for this frame and person
                    frame_data = {
                        'frame': frame_count,
                        'timestamp_ms': frame_timestamp_ms,
                        'person_id': person_idx
                    }
                    
                    # Add each landmark's coordinates (33 landmarks total)
                    # x, y are normalized (0-1), z is depth, visibility is confidence score
                    for landmark_idx, landmark in enumerate(pose_landmarks):
                        frame_data[f'landmark_{landmark_idx}_x'] = landmark.x
                        frame_data[f'landmark_{landmark_idx}_y'] = landmark.y
                        frame_data[f'landmark_{landmark_idx}_z'] = landmark.z
                        frame_data[f'landmark_{landmark_idx}_visibility'] = landmark.visibility
                    
                    pose_data.append(frame_data)
            
            frame_count += 1
            pbar.update(1)
    
    # Release video resources
    cap.release()
    out.release()
    
    # Save pose data to CSV
    if pose_data:
        df = pd.DataFrame(pose_data)
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"\nPose metrics saved to: {CSV_OUTPUT_PATH}")
        print(f"Total records: {len(df)}")
    else:
        print("\nNo pose landmarks detected in the video.")
    
    print(f"Annotated video saved to: {VIDEO_OUTPUT_PATH}")
    print("Processing complete!")
