# Smart Security Camera with Motion Detection and Face Recognition

This project implements a smart security camera system using Python, OpenCV, and Flask. The system detects motion, recognizes faces, sends email alerts, stores footage, and provides a web dashboard for live monitoring.

## 📌 Features

- 🎥 Real-time video streaming via webcam  
- 🕵️‍♂️ Motion detection using frame differencing  
- 🧠 Face recognition with known face database  
- 📧 Email alerts on unknown face detection  
- ☁️ Video and snapshot upload to Google Drive  
- 🌐 Flask-based web dashboard for live monitoring  
- 🔔 Audio alerts for unknown faces  

## 📁 Project Structure

```
smart-security-camera/
├── audio_alerts/
│   └── motion.mp3
│   └── recognized.mp3
│   └── siren.mp3
├── known_faces/
│   └── your_image.jpg      
├── recordings/
│   └── saved_clips.mp4     
├── .env                 
├── classifier.xml              
├── credentials.js          
├── token.js         
        
           
```

## 🔧 Requirements

- Python 3.8+
- OpenCV
- Flask
- face_recognition
- smtplib
- Google Drive API (pydrive)




## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/NskAbdul/smart-security-camera.git
   cd smart-security-camera
   ```

2. Add your known face images in the `known_faces/` folder.

3. Set up email credentials and Google Drive authentication in the appropriate script files.

4. Run the Flask app:
   ```bash
   python smart_camera.py

### ✅ Create and Activate Virtual Environment (Recommended)

To isolate dependencies and avoid conflicts, create a virtual environment:

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 🛡️ Security Features

- Captures and records only when motion is detected.
- Alerts users by email when an unrecognized face is detected.
- Stores evidence securely on the cloud.

## 📬 Email Alert Setup

Configure `.env` with your email credentials:
```python
EMAIL_ADDRESS="your_email@gmail.com"
EMAIL_PASSWORD="your_app_password"
RECIPIENT_EMAIL="your_email@gmail.com"
```


## ☁️ Google Drive Upload

Set up Google Drive API credentials (`credentials.json`) and place them in the project root.
