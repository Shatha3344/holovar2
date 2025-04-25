import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import json
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.environ.get("SESSION_SECRET", "football-analysis-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Create a second static folder for results
from flask import send_from_directory

@app.route('/results/<path:filename>')
def results_files(filename):
    return send_from_directory('results', filename)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'mkv'}

# Make sure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, 'frames'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, 'events'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Sample data flag
USE_SAMPLE_DATA = True  # Set to False in production

# Initialize database for future use if needed
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///football_analysis.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
db.init_app(app)

# Import video processing modules
from video_processor import process_video
from pose_extractor import extract_poses
from event_classifier import classify_events

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Check if we need to generate sample data
    if USE_SAMPLE_DATA and not os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], 'sample_results.json')):
        try:
            import create_sample_content
            create_sample_content.create_sample_data()
            logger.info("Created sample data for demonstration purposes")
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video (this will be a long operation, consider using a task queue in production)
        try:
            logger.info(f"Starting to process video: {filename}")
            
            # Step 1: Process video to extract frames
            frames_path = os.path.join(app.config['RESULTS_FOLDER'], 'frames', timestamp)
            os.makedirs(frames_path, exist_ok=True)
            total_frames, fps = process_video(filepath, frames_path)
            
            # Step 2: Extract poses from frames
            poses_data = extract_poses(frames_path, total_frames)
            
            # Step 3: Classify events based on poses
            events_path = os.path.join(app.config['RESULTS_FOLDER'], 'events', timestamp)
            os.makedirs(events_path, exist_ok=True)
            events = classify_events(poses_data, frames_path, events_path, filename, fps, filepath)
            
            # Save result information
            result_info = {
                'timestamp': timestamp,
                'filename': filename,
                'events_count': len(events),
                'events': events
            }
            
            result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_results.json")
            with open(result_file, 'w') as f:
                json.dump(result_info, f, indent=4)
            
            logger.info(f"Completed processing video: {filename}")
            return redirect(url_for('show_results', timestamp=timestamp))
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            flash(f"Error processing video: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload .mp4 or .mkv files.')
        return redirect(url_for('index'))

@app.route('/sample')
def sample_results():
    """Show sample results for demonstration"""
    return show_results('sample')

@app.route('/results/<timestamp>')
def show_results(timestamp):
    # Check if timestamp is 'sample'
    if timestamp == 'sample':
        result_file = os.path.join(app.config['RESULTS_FOLDER'], "sample_results.json")
    else:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_results.json")
    
    if not os.path.exists(result_file):
        flash('Results not found')
        return redirect(url_for('index'))
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Prepare events data for display
    events_data = []
    for event in results['events']:
        # Get relative path to event image
        image_path = f"results/events/{timestamp}/{event['frame_number']}_{event['label']}.jpg"
        
        event_data = {
            'type': event['label'],
            'time': event['game_time'],
            'frame': event['frame_number'],
            'image_path': image_path
        }
        
        # Add clip path if available
        if 'clip_path' in event:
            event_data['clip_path'] = f"results/events/{timestamp}/{event['clip_path']}"
            
        events_data.append(event_data)
    
    return render_template('results.html', 
                          filename=results['filename'],
                          timestamp=timestamp,
                          events_count=results['events_count'],
                          events=events_data)

@app.route('/api/events/<timestamp>')
def get_events(timestamp):
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_results.json")
    
    if not os.path.exists(result_file):
        return jsonify({'error': 'Results not found'}), 404
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
