import os
import logging
import numpy as np
import json
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from video_processor import convert_frame_to_time
from pose_extractor import create_temporal_sequences

logger = logging.getLogger(__name__)

# Event classes 
EVENT_CLASSES = ['Goal', 'Foul', 'Offside', 'Corner', 'Save', 'Tackle', 'Pass', 'None']

class EventClassifier:
    def __init__(self):
        """
        Initialize the LSTM model for event classification
        
        In a production environment, this would load a pre-trained model
        Here we're creating a simple model architecture for demonstration
        """
        self.model = self._create_model()
        
        # For demonstration, we'll use a simple rule-based classifier
        # In production, this should be replaced with a trained model
        logger.warning("Using rule-based classifier as a placeholder for LSTM model")
    
    def _create_model(self):
        """
        Create LSTM model for sequence classification
        
        Returns:
            keras.Model: LSTM model
        """
        # Input shape: (sequence_length, num_keypoints * 3)
        # 33 keypoints with x, y, z coordinates
        input_shape = (30, 33 * 3)
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(EVENT_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_sequence(self, sequence):
        """
        Preprocess a sequence of poses for the model
        
        Args:
            sequence (list): List of pose keypoints for multiple frames
            
        Returns:
            numpy.ndarray: Preprocessed sequence
        """
        # Flatten each pose (33 keypoints * 3 coordinates)
        flattened = []
        for pose in sequence:
            flat_pose = np.array(pose).flatten()
            flattened.append(flat_pose)
        
        # Convert to numpy array
        return np.array([flattened])
    
    def classify(self, sequence):
        """
        Classify a sequence of poses
        
        Args:
            sequence (list): List of pose keypoints for multiple frames
            
        Returns:
            str: Event class
        """
        # In a production environment, this would use the trained model
        # For demonstration, we're using a simple rule-based classifier
        
        # Get the starting and ending frames' poses
        start_pose = np.array(sequence[0])
        end_pose = np.array(sequence[-1])
        
        # Calculate the average movement of all keypoints
        movement = np.mean(np.abs(end_pose - start_pose))
        
        # Rule-based classification using a wider set of events
        all_events = [
            'Ball out of play', 'Clearance', 'Close-up behind the goal', 'Close-up corner',
            'Close-up player or field referee', 'Close-up side staff', 'Corner', 'Direct free-kick',
            'Foul', 'Goal', 'Goal line technology camera', 'Indirect free-kick', 'Inside the goal',
            'Kick-off', 'Main behind the goal', 'Main camera center', 'Main camera left',
            'Main camera right', 'Offside', 'Other', 'Penalty', 'Public', 'Red card',
            'Shots off target', 'Shots on target', 'Spider camera', 'Substitution',
            'Throw-in', 'Yellow card', 'Yellow->red card'
        ]
        
        # For demo purposes, deterministically select events based on movement level
        if movement > 0.2:
            # High movement events
            high_movement_events = ['Goal', 'Penalty', 'Shots on target', 'Shots off target', 'Foul', 'Red card']
            index = int(movement * 100) % len(high_movement_events)
            return high_movement_events[index]
        elif 0.15 < movement <= 0.2:
            # Medium-high movement events
            medium_high_events = ['Corner', 'Direct free-kick', 'Indirect free-kick', 'Yellow card', 'Yellow->red card']
            index = int(movement * 100) % len(medium_high_events)
            return medium_high_events[index]
        elif 0.08 < movement <= 0.15:
            # Medium movement events
            medium_events = ['Clearance', 'Offside', 'Throw-in', 'Ball out of play', 'Kick-off', 'Substitution']
            index = int(movement * 100) % len(medium_events)
            return medium_events[index]
        elif 0.03 < movement <= 0.08:
            # Low movement events - typically camera angles and close-ups
            low_events = ['Close-up corner', 'Close-up player or field referee', 'Close-up side staff', 
                         'Main camera center', 'Main camera left', 'Main camera right', 'Public']
            index = int(movement * 100) % len(low_events)
            return low_events[index]
        else:
            # Very low movement - probably special camera shots
            special_events = ['Inside the goal', 'Goal line technology camera', 'Spider camera', 
                             'Close-up behind the goal', 'Main behind the goal', 'Other']
            index = int(movement * 1000) % len(special_events)
            return special_events[index]


def classify_events(poses_data, frames_folder, output_folder, video_filename, fps=30, video_path=None):
    """
    Classify events based on pose data
    
    Args:
        poses_data (dict): Dictionary mapping frame number to pose data
        frames_folder (str): Folder containing the frames
        output_folder (str): Folder to save event data and images
        video_filename (str): Original video filename
        fps (float): Frames per second
        video_path (str, optional): Path to the original video file for event clip extraction
        
    Returns:
        list: List of events
    """
    logger.info("Classifying events...")
    
    # Check if we have enough pose data
    if len(poses_data) < 5:
        logger.warning("Not enough pose data for proper sequence analysis. Creating minimal demo events.")
        # For demo, let's create some example events directly
        events = []
        
        # Event types to demonstrate
        event_types = ["Goal", "Foul", "Offside", "Corner", "Save"]
        
        # Get available frame numbers
        frame_numbers = sorted(list(poses_data.keys()))
        
        # If we have frames, create events for them
        if frame_numbers:
            # Distribute events across available frames
            for i, event_type in enumerate(event_types):
                if i < len(frame_numbers):
                    frame_num = frame_numbers[i]
                    
                    # Generate game time
                    game_time = convert_frame_to_time(frame_num, fps)
                    
                    # Create event data
                    event_data = {
                        "match_id": os.path.splitext(video_filename)[0],
                        "video_file": video_filename,
                        "camera_view": "Main",
                        "frame_number": frame_num,
                        "game_time": game_time,
                        "label": event_type,
                        "pose_3d": poses_data[frame_num]
                    }
                    
                    # Save event image with label
                    frame_path = os.path.join(frames_folder, f"frame_{frame_num:06d}.jpg")
                    if os.path.exists(frame_path):
                        event_image_path = os.path.join(output_folder, f"{frame_num}_{event_type}.jpg")
                        frame = cv2.imread(frame_path)
                        
                        # Add event label to the image
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
                        
                        cv2.imwrite(event_image_path, frame)
                    
                    # Extract video clip if original video path is available
                    if video_path:
                        from video_processor import extract_event_video
                        clip_path = extract_event_video(video_path, frame_num, fps, output_folder)
                        if clip_path:
                            event_data["clip_path"] = os.path.basename(clip_path)

                    # Save event JSON
                    event_json_path = os.path.join(output_folder, f"{frame_num}_{event_type}.json")
                    with open(event_json_path, 'w') as f:
                        json.dump(event_data, f, indent=4)
                    
                    events.append(event_data)
        
        logger.info(f"Created {len(events)} demo events")
        return events
        
    # If we have enough data, proceed with normal classification
    # Create sequences from pose data
    sequences = create_temporal_sequences(poses_data, sequence_length=5)  # Shorter sequences for demo
    
    # Initialize classifier
    classifier = EventClassifier()
    
    events = []
    
    # Limit the number of sequences for demo purposes
    max_sequences = min(len(sequences), 10)
    actual_sequences = sequences[:max_sequences]
    
    for i, seq in enumerate(actual_sequences):
        start_frame, end_frame = seq['frame_range']
        
        # Classify sequence
        label = classifier.classify(seq['poses'])
        
        # Skip if no event detected
        if label == 'None':
            continue
        
        # Get middle frame for visualization
        mid_frame = (start_frame + end_frame) // 2
        frame_path = os.path.join(frames_folder, f"frame_{mid_frame:06d}.jpg")
        
        # Generate game time
        game_time = convert_frame_to_time(mid_frame, fps)
        
        # Save event data
        event_data = {
            "match_id": os.path.splitext(video_filename)[0],
            "video_file": video_filename,
            "camera_view": "Main",
            "frame_number": mid_frame,
            "game_time": game_time,
            "label": label,
            "pose_3d": seq['poses'][len(seq['poses'])//2]  # Use middle frame's pose
        }
        
        # Save event image with label
        if os.path.exists(frame_path):
            event_image_path = os.path.join(output_folder, f"{mid_frame}_{label}.jpg")
            frame = cv2.imread(frame_path)
            
            # Add event label to the image
            cv2.putText(
                frame, 
                f"{label} - {game_time}", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
            
            cv2.imwrite(event_image_path, frame)
        else:
            # If the specific frame doesn't exist, use any available frame for demonstration
            existing_frames = [f for f in os.listdir(frames_folder) if f.startswith('frame_') and f.endswith('.jpg')]
            if existing_frames:
                sample_frame_path = os.path.join(frames_folder, existing_frames[0])
                event_image_path = os.path.join(output_folder, f"{mid_frame}_{label}.jpg")
                frame = cv2.imread(sample_frame_path)
                
                # Add event label to the image
                cv2.putText(
                    frame, 
                    f"{label} - {game_time} (Demo Frame)", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA
                )
                
                cv2.imwrite(event_image_path, frame)
        
        # Extract video clip if original video path is available
        if video_path:
            from video_processor import extract_event_video
            clip_path = extract_event_video(video_path, mid_frame, fps, output_folder)
            if clip_path:
                event_data["clip_path"] = os.path.basename(clip_path)
        
        # Save event JSON
        event_json_path = os.path.join(output_folder, f"{mid_frame}_{label}.json")
        with open(event_json_path, 'w') as f:
            json.dump(event_data, f, indent=4)
        
        events.append(event_data)
    
    logger.info(f"Classified {len(events)} events")
    return events
