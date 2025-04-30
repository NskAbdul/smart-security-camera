import os
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import time
from dotenv import load_dotenv
import pygame
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'camera_width': 1280,
    'camera_height': 720,
    'motion_sensitivity': 500,
    'min_recording_time': 10,
    'motion_cooldown': 2,
    'alert_cooldown': 60,
    'recordings_dir': "recordings",
    'audio_alerts_dir': "audio_alerts",
    'known_faces_dir': "known_faces",
    'training_data_dir': "data",
    'classifier_file': "classifier.xml"
}

# Global state
state = {
    'motion_detected': False,
    'last_motion_time': 0,
    'video_writer': None,
    'recording_start_time': 0,
    'camera': None,
    'last_alert_time': 0,
    'background_subtractor': cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False),
    'email_sent': False,
    'video_uploaded': False,
    'face_cascade': cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    'face_recognizer': None,
    'audio_files': {
        'motion': None,
        'recognized': None,
        'siren': None
    }
}

def initialize_directories():
    """Create all required directories if they don't exist"""
    os.makedirs(CONFIG['recordings_dir'], exist_ok=True)
    os.makedirs(CONFIG['known_faces_dir'], exist_ok=True)
    os.makedirs(CONFIG['audio_alerts_dir'], exist_ok=True)
    os.makedirs(CONFIG['training_data_dir'], exist_ok=True)
    print(f"Created directories: {CONFIG['recordings_dir']}, {CONFIG['known_faces_dir']}, {CONFIG['audio_alerts_dir']}, {CONFIG['training_data_dir']}")

def load_audio_files():
    """Load all audio files into memory"""
    try:
        state['audio_files']['motion'] = pygame.mixer.Sound(os.path.join(CONFIG['audio_alerts_dir'], "motion.mp3"))
        state['audio_files']['recognized'] = pygame.mixer.Sound(os.path.join(CONFIG['audio_alerts_dir'], "recognized.mp3"))
        state['audio_files']['siren'] = pygame.mixer.Sound(os.path.join(CONFIG['audio_alerts_dir'], "siren.mp3"))
        print("Audio files loaded successfully")
    except Exception as e:
        print(f"Failed to load audio files: {e}")
        # Create dummy sound objects if files couldn't be loaded
        silent_sound = pygame.mixer.Sound(buffer=bytearray(100))
        state['audio_files']['motion'] = silent_sound
        state['audio_files']['recognized'] = silent_sound
        state['audio_files']['siren'] = silent_sound

def initialize_camera():
    """Initialize and configure the camera"""
    try:
        state['camera'] = cv2.VideoCapture(0)
        if not state['camera'].isOpened():
            raise RuntimeError("Cannot open camera")
        
        state['camera'].set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
        state['camera'].set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
        
        initialize_directories()
        load_audio_files()
        initialize_face_recognition()
        
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        raise

def initialize_face_recognition():
    """Initialize face recognition system"""
    try:
        # Load or train the face recognizer
        if os.path.exists(CONFIG['classifier_file']):
            state['face_recognizer'] = cv2.face.LBPHFaceRecognizer_create()
            state['face_recognizer'].read(CONFIG['classifier_file'])
            print("Loaded trained face recognizer")
        else:
            print("No trained model found. Training new model...")
            train_classifier()
            
    except Exception as e:
        print(f"Face recognition initialization failed: {e}")
        raise

def train_classifier():
    """Train the face recognition classifier using images from the data directory"""
    print("\nTraining face recognition model...")
    try:
        # Get the training data
        faces = []
        ids = []
        
        # Loop through all training images
        for root, dirs, files in os.walk(CONFIG['training_data_dir']):
            for file in files:
                if file.startswith("user.") and file.endswith(".jpg"):
                    # Extract ID from filename (format: user.<id>.<number>.jpg)
                    try:
                        id = int(file.split(".")[1])
                        if id not in [1, 2, 3]:  # Only accept IDs 1, 2, or 3
                            continue
                            
                        img_path = os.path.join(root, file)
                        
                        # Convert to grayscale
                        img = Image.open(img_path).convert('L')
                        img_np = np.array(img, 'uint8')
                        
                        # Detect face in the image
                        detected_faces = state['face_cascade'].detectMultiScale(img_np)
                        if len(detected_faces) == 1:
                            (x, y, w, h) = detected_faces[0]
                            face_img = img_np[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (200, 200))
                            faces.append(face_img)
                            ids.append(id)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
        
        if len(faces) == 0:
            raise ValueError("No valid training images found")
        
        ids = np.array(ids)
        
        # Train and save classifier
        state['face_recognizer'] = cv2.face.LBPHFaceRecognizer_create()
        state['face_recognizer'].train(faces, ids)
        state['face_recognizer'].write(CONFIG['classifier_file'])
        print(f"Training completed. Model saved to {CONFIG['classifier_file']}")
        print(f"Trained with {len(faces)} images for {len(np.unique(ids))} persons")
        
        # Print training summary
        unique_ids, counts = np.unique(ids, return_counts=True)
        for id, count in zip(unique_ids, counts):
            print(f" - ID {id}: {count} images")
        
    except Exception as e:
        print(f"Failed to train classifier: {e}")

def generate_dataset():
    """Generate training dataset by capturing face images from webcam for multiple users"""
    print("\nStarting face image collection...")
    try:
        # Get user IDs and names to collect
        users = {
            1: "Abdul",
            2: "Karthikeya",
            3: "Third_Person"  # Replace with actual name
        }
        
        for user_id, user_name in users.items():
            print(f"\nCollecting images for {user_name} (ID: {user_id})")
            img_count = 0
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = state['face_cascade'].detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    img_count += 1
                    
                    # Save the captured face image
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))
                    file_path = os.path.join(CONFIG['training_data_dir'], f"user.{user_id}.{img_count}.jpg")
                    cv2.imwrite(file_path, face_img)
                    
                    cv2.putText(frame, f"{user_name}: {img_count}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(f"Collecting Face Data for {user_name}", frame)
                
                if cv2.waitKey(1) == 13 or img_count >= 100:  # Enter key or 100 images
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"Collected {img_count} samples for {user_name}")
        
        # Retrain the model with new data
        train_classifier()
        
    except Exception as e:
        print(f"Dataset generation failed: {e}")

def recognize_face(face_img):
    """Recognize face using the trained model"""
    if state['face_recognizer'] is None:
        return "Unknown", 0
    
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (200, 200))
    id, confidence = state['face_recognizer'].predict(gray_face)
    confidence = int(100 * (1 - confidence / 300))
    
    # Map IDs to names
    id_to_name = {
        1: "Abdul",
        2: "Karthikeya",
        3: "Third_Person"  # Replace with actual name
    }
    
    if confidence > 70 and id in id_to_name:
        return id_to_name[id], confidence
    return "Unknown", confidence
def detect_motion_and_faces(frame):
    """Detect motion and recognize faces"""
    motion_detected = False
    recognized_faces = []
    motion_boxes = []
    current_name = "Unknown"
    
    try:
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = state['background_subtractor'].apply(gray)
        
        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > CONFIG['motion_sensitivity']:
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))
                motion_detected = True
        
        # Face recognition
        if motion_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = state['face_cascade'].detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                name, confidence = recognize_face(face_img)
                recognized_faces.append(name)
                current_name = name
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} {confidence}%", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Logging functionality
        if motion_detected:
            print("[INFO] Motion Detected")

            if current_name == "Unknown":
                print("[INFO] Face not recognized: Unauthorized entry")
                
                if not state['email_sent']:
                    print("[ACTION] Preparing to send email to owner")
                    state['email_sent'] = True

                if not state['video_uploaded']:
                    print("[ACTION] Preparing to upload video to Google Drive")
                    state['video_uploaded'] = True
            
            else:
                print(f"[INFO] Known person detected: {current_name}")
                print("[INFO] No email or upload triggered for known person")
    
    except Exception as e:
        print(f"Detection error: {e}")
    
    return motion_detected, recognized_faces, motion_boxes

def start_recording():
    """Start recording video"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CONFIG['recordings_dir'], f"recording_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0
    frame_size = (CONFIG['camera_width'], CONFIG['camera_height'])
    
    state['video_writer'] = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    state['recording_start_time'] = time.time()
    state['email_sent'] = False
    state['video_uploaded'] = False
    
    print(f"Started recording: {filename}")

def stop_recording():
    """Stop recording video"""
    if state['video_writer'] is not None:
        state['video_writer'].release()
        state['video_writer'] = None
        print("Stopped recording")
        
        # Get the latest recording file
        recordings = sorted(os.listdir(CONFIG['recordings_dir']))
        if recordings:
            latest_recording = os.path.join(CONFIG['recordings_dir'], recordings[-1])
            
            # Send email and upload to drive if unknown person detected
            if state['email_sent']:
                send_email(latest_recording)
                upload_to_drive(latest_recording)

def handle_recording(motion_detected, frame):
    """Handle recording based on motion detection"""
    current_time = time.time()
    
    if motion_detected:
        state['last_motion_time'] = current_time
        
        if state['video_writer'] is None:
            start_recording()
    
    if state['video_writer'] is not None:
        # Continue recording for minimum time or while motion is detected
        recording_duration = current_time - state['recording_start_time']
        
        if (current_time - state['last_motion_time'] > CONFIG['motion_cooldown'] and 
            recording_duration >= CONFIG['min_recording_time']):
            stop_recording()
        else:
            # Write the current frame to video
            state['video_writer'].write(frame)

def play_audio_alert(alert_type):
    """Play audio alert based on the alert type"""
    current_time = time.time()
    
    if current_time - state['last_alert_time'] > CONFIG['alert_cooldown']:
        try:
            if alert_type in state['audio_files'] and state['audio_files'][alert_type]:
                # Stop any currently playing sound
                pygame.mixer.stop()
                # Play the new sound
                state['audio_files'][alert_type].play()
                state['last_alert_time'] = current_time
                print(f"[AUDIO] Playing {alert_type} alert")
        except Exception as e:
            print(f"Failed to play audio alert: {e}")

def upload_to_drive(file_path):
    """Upload file to Google Drive"""
    try:
        print(f"Uploading {file_path} to Google Drive...")
        
        # Authenticate and create service
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', ['https://www.googleapis.com/auth/drive.file'])
        creds = flow.run_local_server(port=0)
        
        service = build('drive', 'v3', credentials=creds)
        
        # Upload file
        file_metadata = {
            'name': os.path.basename(file_path),
            'mimeType': 'video/mp4'
        }
        
        media = MediaFileUpload(file_path, mimetype='video/mp4')
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        print(f"Uploaded file ID: {file.get('id')}")
        return True
        
    except Exception as e:
        print(f"Failed to upload to Google Drive: {e}")
        return False

def send_email(attachment_path=None):
    """Send email notification with optional attachment"""
    try:
        print("Preparing email...")
        
        # Email configuration
        sender_email = os.getenv('EMAIL_USER', 'karthiksiri2906@gmail.com')
        sender_password = os.getenv('EMAIL_PASSWORD', 'efan rlpa ydle vufe')
        recipient_email = os.getenv('RECIPIENT_EMAIL', 'nsk.abdul786@gmail.com')
        
        if not sender_email or not sender_password:
            print("Email credentials not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Security Alert: Motion Detected"
        
        body = "Motion was detected by your security camera. Please check the attached recording."
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach file if provided
        if attachment_path:
            attachment = open(attachment_path, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 
                           f"attachment; filename= {os.path.basename(attachment_path)}")
            msg.attach(part)
            attachment.close()
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print("Email sent successfully")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def add_visualizations(frame, motion_boxes):
    """Add visualizations to the frame"""
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add motion boxes
    for (x, y, w, h) in motion_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Add recording indicator
    if state['video_writer'] is not None:
        cv2.putText(frame, "REC", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def camera_loop():
    """Main camera processing loop"""
    try:
        while True:
            ret, frame = state['camera'].read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            motion_detected, recognized_faces, motion_boxes = detect_motion_and_faces(frame)
            state['motion_detected'] = motion_detected
            
            # Handle recording with the current frame
            handle_recording(motion_detected, frame)
            
            # Play appropriate audio alert
            if motion_detected:
                if "Unknown" in recognized_faces:
                    play_audio_alert("siren")
                elif recognized_faces:  # If any recognized faces
                    play_audio_alert("recognized")
                else:  # Just motion, no faces
                    play_audio_alert("motion")
            
            # Add visualizations
            frame = add_visualizations(frame, motion_boxes)
            
            # Display the frame in a window
            cv2.imshow('Security Camera', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Camera loop error: {e}")
    finally:
        # Ensure recording is stopped before exiting
        if state['video_writer'] is not None:
            stop_recording()
        if state['camera'] is not None:
            state['camera'].release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == '__main__':
    try:
        # Initialize components
        initialize_camera()
        
        # Check if we need to collect training data
        training_files = os.listdir(CONFIG['training_data_dir'])
        user_ids = set()
        
        for file in training_files:
            if file.startswith("user.") and file.endswith(".jpg"):
                try:
                    user_id = int(file.split(".")[1])
                    user_ids.add(user_id)
                except:
                    continue
        
        if len(user_ids) < 2:  # We need at least 2 users
            print("Incomplete training data found. Starting dataset collection...")
            generate_dataset()
        
        # Start camera processing
        print("\nStarting camera...")
        camera_loop()
        
    except Exception as e:
        print(f"Application failed: {e}")
    finally:
        print("Application shutdown")
        if state['camera'] is not None:
            state['camera'].release()
        cv2.destroyAllWindows()