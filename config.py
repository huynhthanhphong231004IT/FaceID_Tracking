from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
CSV_PATH = DATABASE_DIR / "face_features.csv"          

WEIGHTS_DIR = BASE_DIR / "models" / "weights"
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n-face.pt"    
YOLO_WEIGHTS_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
INSIGHTFACE_ROOT = WEIGHTS_DIR / "insightface"
INSIGHTFACE_MODEL = "buffalo_l"                         

CAMERA_ID = 0                    
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FRAME_QUEUE_SIZE = 2             
CAMERA_MAX_READ_FAILURES = 30    
DEVICE = "auto"
YOLO_CONF = 0.5                
YOLO_IOU = 0.5                   
YOLO_IMGSZ = 640                 
YOLO_MAX_DET = 20                
LANDMARK_FALLBACK_SIZE = (192, 192)   
LANDMARK_CROP_MARGIN = 0.40           
LANDMARK_DET_THRESH = 0.30

ALIGN_SIZE = 112               
MIN_FACE_SIZE = 48               
POSE_MAX_YAW = 20.0              
POSE_MAX_PITCH = 20.0            
POSE_MAX_ROLL = 20.0            
BLUR_THRESHOLD = 35.0            

EMBEDDING_DIM = 512
MATCH_DISTANCE_THRESHOLD = 0.45  
UPDATE_DISTANCE_THRESHOLD = 0.30 
N_CONFIRM_FRAMES = 5             
FEATURE_UPDATE_MIN_INTERVAL = 1.0  
MAX_FEATURES_PER_PERSON = 100    

TRACK_IOU_THRESHOLD = 0.30       
TRACK_MAX_MISSED = 20            

ENROLL_SAMPLES = 12              
ENROLL_MIN_INTERVAL = 0.20       
ENROLL_TIMEOUT_SEC = 60         
ENROLL_REJECT_DUPLICATE = True  

AUTO_ENROLL_ENABLED = True
AUTO_ENROLL_MIN_SAMPLES = 20         
AUTO_ENROLL_MIN_SECONDS = 3.0        
AUTO_ENROLL_SAMPLE_INTERVAL = 0.15   
AUTO_ENROLL_COHERENCE_MEAN = 0.60    
AUTO_ENROLL_COHERENCE_MIN = 0.40    
AUTO_ENROLL_MIN_DB_DISTANCE = 0.55  
                                     
AUTO_ENROLL_MIN_FACE_SIZE = 80       
AUTO_ENROLL_CANDIDATE_TTL = 20.0     
AUTO_ENROLL_MAX_CANDIDATES = 8       
AUTO_NAME_PREFIX = "Person_"        

DB_AUTOSAVE_INTERVAL = 10.0      

APP_TITLE = "FACE - TRACKER // Real-time Re-ID with YOLO + ArcFace"
UI_REFRESH_MS = 15               
LOG_LEVEL = "INFO"

COLORS = {
    "bg": "#070b14",
    "panel": "#0d1424",
    "panel_alt": "#121c33",
    "border": "#1e2d4f",
    "cyan": "#00e5ff",
    "magenta": "#ff2d95",
    "lime": "#39ff88",
    "amber": "#ffb300",
    "red": "#ff3b5c",
    "text": "#d6e4ff",
    "muted": "#6b7fa6",
}
