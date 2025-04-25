import os
import cv2
import numpy as np
import json

# Create sample images and data for demo
def create_sample_data():
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results/frames/sample', exist_ok=True)
    os.makedirs('results/events/sample', exist_ok=True)
    
    # Create some sample frames
    for i in range(10):
        # Create a black image with text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a football field background
        # Green field
        cv2.rectangle(img, (0, 0), (640, 480), (0, 120, 0), -1)
        
        # Field lines
        cv2.rectangle(img, (50, 50), (590, 430), (255, 255, 255), 2)
        cv2.line(img, (320, 50), (320, 430), (255, 255, 255), 2)
        cv2.circle(img, (320, 240), 50, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(
            img,
            f"Frame {i}",
            (50, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Add some players (as colored circles)
        # Team 1 (blue)
        for _ in range(5):
            x = np.random.randint(100, 540)
            y = np.random.randint(100, 380)
            cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
        
        # Team 2 (red)
        for _ in range(5):
            x = np.random.randint(100, 540)
            y = np.random.randint(100, 380)
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        
        # Ball (white)
        ball_x = np.random.randint(100, 540)
        ball_y = np.random.randint(100, 380)
        cv2.circle(img, (ball_x, ball_y), 5, (255, 255, 255), -1)
        
        # Save the image
        cv2.imwrite(f'results/frames/sample/frame_{i:06d}.jpg', img)
    
    # Create sample events using the expanded list
    event_types = ["Goal", "Penalty", "Red card", "Yellow card", "Corner", 
                  "Offside", "Throw-in", "Shots on target", "Direct free-kick", 
                  "Close-up player or field referee", "Main camera center"]
    
    for i, event_type in enumerate(event_types):
        if i < 5:  # Use 5 frames for 5 event types
            # Create a copy of the frame with event text
            frame = cv2.imread(f'results/frames/sample/frame_{i:06d}.jpg')
            
            # Add event label
            game_time = f"{i//2:02d}:{(i*10)%60:02d}"
            cv2.putText(
                frame,
                f"{event_type} - {game_time}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Save labeled image
            cv2.imwrite(f'results/events/sample/{i}_{event_type}.jpg', frame)
            
            # Create sample 3D pose data (33 keypoints with x,y,z coordinates)
            pose_3d = []
            for _ in range(33):
                point = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                pose_3d.append(point)
            
            # Create event JSON
            event_data = {
                "match_id": "sample_match",
                "video_file": "sample_video.mp4",
                "camera_view": "Main",
                "frame_number": i,
                "game_time": game_time,
                "label": event_type,
                "pose_3d": pose_3d
            }
            
            # Save event JSON
            with open(f'results/events/sample/{i}_{event_type}.json', 'w') as f:
                json.dump(event_data, f, indent=4)
    
    # Create a results JSON file
    results = {
        "timestamp": "sample",
        "filename": "sample_video.mp4",
        "events_count": len(event_types),
        "events": [
            {
                "match_id": "sample_match",
                "video_file": "sample_video.mp4",
                "camera_view": "Main",
                "frame_number": i,
                "game_time": f"{i//2:02d}:{(i*10)%60:02d}",
                "label": event_type
            } for i, event_type in enumerate(event_types) if i < 5
        ]
    }
    
    with open(f'results/sample_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Sample data created successfully!")

if __name__ == "__main__":
    create_sample_data()