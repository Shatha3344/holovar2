import cv2
import os
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def process_video(video_path, output_folder):
    """
    Process a football match video, extract frames and save them
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames
        
    Returns:
        int: Total number of frames processed
    """
    logger.info(f"Processing video: {video_path}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    logger.info(f"Video FPS: {fps}")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    # We'll extract 1 frame every 2 seconds to reduce processing time
    # In production, you might want to process all frames or at a higher sampling rate
    frame_interval = int(fps) * 2  # 1 frame every 2 seconds
    
    frames_processed = 0
    frame_index = 0
    
    # Use tqdm for progress bar
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every N frames (1 per second)
            if frame_index % frame_interval == 0:
                # Save frame as image
                frame_filename = os.path.join(output_folder, f"frame_{frames_processed:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frames_processed += 1
            
            frame_index += 1
            pbar.update(1)
    
    cap.release()
    logger.info(f"Processed {frames_processed} frames from video")
    
    return frames_processed, fps

def extract_frame_at_time(video_path, time_seconds, output_path):
    """
    Extract a specific frame at a given time from a video
    
    Args:
        video_path (str): Path to the video file
        time_seconds (float): Time in seconds to extract the frame from
        output_path (str): Path to save the extracted frame
        
    Returns:
        bool: True if frame was successfully extracted, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame number from time
    frame_number = int(time_seconds * fps)
    
    # Set video to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        logger.error(f"Could not read frame at time {time_seconds}s")
        cap.release()
        return False
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    cap.release()
    
    return True

def convert_frame_to_time(frame_number, fps):
    """
    Convert a frame number to time format (MM:SS)
    
    Args:
        frame_number (int): Frame number
        fps (float): Frames per second
        
    Returns:
        str: Time in format MM:SS
    """
    total_seconds = int(frame_number / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    return f"{minutes:02d}:{seconds:02d}"

def extract_video_segment(video_path, start_time, end_time, output_path):
    """
    Extract a segment of a video between start_time and end_time
    
    Args:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_path (str): Path to save the extracted video segment
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Calculate start and end frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract and write frames
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True
    except Exception as e:
        logger.error(f"Error extracting video segment: {str(e)}")
        return False

def extract_event_video(video_path, event_frame, fps, output_folder, seconds_before=5, seconds_after=5):
    """
    Extract a video segment around an event
    
    Args:
        video_path (str): Path to the video file
        event_frame (int): Frame number of the event
        fps (float): Frames per second of the video
        output_folder (str): Folder to save the event video
        seconds_before (int): Number of seconds to include before the event
        seconds_after (int): Number of seconds to include after the event
    
    Returns:
        str: Path to the extracted video segment, or None if extraction failed
    """
    # Calculate event time in seconds
    event_time = event_frame / fps
    
    # Calculate start and end times
    start_time = max(0, event_time - seconds_before)
    end_time = event_time + seconds_after
    
    # Create output filename
    output_filename = f"event_{event_frame:06d}.mp4"
    output_path = os.path.join(output_folder, output_filename)
    
    # Extract the video segment
    if extract_video_segment(video_path, start_time, end_time, output_path):
        return output_path
    
    return None
