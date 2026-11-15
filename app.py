import io
import time
import cv2
import numpy as np
from jose import jwt
import uuid

from datetime import datetime, timedelta
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy.orm import Session
from database import get_db
from models import User, Prediction

from passlib.context import CryptContext
from schemas import RegisterRequest, LoginRequest
from models import User
security = HTTPBearer()
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token."
            )

        user = db.query(User).filter(
            User.id == int(user_id)
        ).first()

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User not found."
            )

        return user

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token."
        )

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)
SECRET_KEY = "change-this-to-a-random-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt

app = FastAPI(
    title="Diabetic Retinopathy Screening API",
    description="Edge screening API powered by distilled MobileNetV3 with calibrated decision cutoffs."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Deployment Artifacts
ONNX_MODEL_PATH = "mobilenetv3_student_diabetic_retinopathy.onnx"
THRESHOLD_PATH = "calibrated_thresholds_unified.npy"

# Clinical stage taxonomy
CLINICAL_STAGES = {
    0: ("No DR", False, "Normal", "Routine annual screening."),
    1: ("Mild Non-Proliferative DR", False, "Low", "Rescreen in 12 months."),
    2: ("Moderate Non-Proliferative DR", True, "Moderate", "Refer to eye clinic within 2-3 months."),
    3: ("Severe Non-Proliferative DR", True, "High", "Refer to ophthalmologist within 2-4 weeks."),
    4: ("Proliferative DR", True, "Critical", "Urgent referral to retinal specialist within 1 week.")
}

# Initialize ONNX Runtime Session
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4

session = ort.InferenceSession(ONNX_MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Load Calibrated Cutoffs
try:
    calibrated_thresholds = np.sort(np.load(THRESHOLD_PATH).astype(np.float32)).tolist()
    print(f"Loaded calibrated thresholds: {np.round(calibrated_thresholds, 4).tolist()}")
except Exception as e:
    raise RuntimeError(f"Could not load '{THRESHOLD_PATH}': {e}")


def preprocess_fundus(image_bytes: bytes, target_size: int = 384) -> np.ndarray:
    """Mirrors the exact training pipeline: Crop -> 384x384 -> Ben Graham -> Green CLAHE -> Norm."""
    # Decode bytes to BGR numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    # 1. Circular Mask & Dynamic Retinal Bounding Crop
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 8
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img_bgr = img_bgr[y0:y1, x0:x1]

    # 2. Rescale to 384x384
    img_resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 3. Ben Graham Local Illumination Subtraction (sigma=10)
    blur = cv2.GaussianBlur(img_resized, (0, 0), sigmaX=10)
    ben_graham = cv2.addWeighted(img_resized, 4, blur, -4, 128)

    # 4. Green-Channel CLAHE (Hemoglobin band isolation)
    b, g, r = cv2.split(ben_graham)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    processed = cv2.merge((b, g_clahe, r))

    # 5. ImageNet Normalization (BGR -> RGB)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std

    # 6. HWC to NCHW format
    tensor = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32)


@app.get("/")
def home():
    return {
        "status": "healthy",
        "model": ONNX_MODEL_PATH,
        "input_resolution": "384x384",
        "active_thresholds": calibrated_thresholds
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "onnx_model": ONNX_MODEL_PATH,
        "input_resolution": "384x384"
    }
@app.get("/model/info")
def model_info():
    return {
        "model_name": "MobileNetV3 Student",
        "model_file": ONNX_MODEL_PATH,
        "format": "ONNX",
        "input_resolution": "384x384",
        "input_name": input_name,
        "classes": 5,
        "class_mapping": {
            "0": "No DR",
            "1": "Mild Non-Proliferative DR",
            "2": "Moderate Non-Proliferative DR",
            "3": "Severe Non-Proliferative DR",
            "4": "Proliferative DR"
        },
        "thresholds": calibrated_thresholds
    }
@app.get("/db-test")
def database_test(db: Session = Depends(get_db)):
    return {
        "database": "connected",
        "status": "SQLite is working"
    }
def check_image_gradability(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "gradable": False,
            "blur_variance": 0.0,
            "mean_illumination": 0.0,
            "blur_pass": False,
            "illumination_pass": False,
            "reasons": ["Invalid image file."]
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    mean_illumination = float(
        np.mean(gray)
    )

    MIN_BLUR = 20.0
    MIN_ILLUMINATION = 20.0
    MAX_ILLUMINATION = 240.0

    blur_pass = blur_variance >= MIN_BLUR

    illumination_pass = (
        MIN_ILLUMINATION
        <= mean_illumination
        <= MAX_ILLUMINATION
    )

    gradable = (
        blur_pass
        and illumination_pass
    )

    reasons = []

    if not blur_pass:
        reasons.append(
            "Image is too blurry for reliable screening."
        )

    if not illumination_pass:
        reasons.append(
            "Image illumination is outside the acceptable range."
        )

    return {
        "gradable": gradable,
        "blur_variance": round(
            blur_variance,
            4
        ),
        "mean_illumination": round(
            mean_illumination,
            4
        ),
        "blur_pass": blur_pass,
        "illumination_pass": illumination_pass,
        "reasons": reasons
    }

@app.post("/screen/gradability")
async def predict_eye(file: UploadFile):
    start_time = time.perf_counter()

    image_bytes = await file.read()

    # Check image quality before AI screening
    gradability = check_image_gradability(image_bytes)

    if not gradability["gradable"]:
        raise HTTPException(
            status_code=400,
            detail={
                "filename": file.filename,
                "message": "Image is not suitable for reliable screening.",
                "reasons": gradability["reasons"],
                "blur_variance": gradability["blur_variance"],
                "mean_illumination": gradability["mean_illumination"]
            }
        )

    try:
        img_tensor = preprocess_fundus(
            image_bytes,
            target_size=384
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format for {file.filename}."
        )

    outputs = session.run(
        None,
        {input_name: img_tensor}
    )

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Make sure the image is valid
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Blur check
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Illumination check
    mean_illumination = float(np.mean(gray))

    # Thresholds from your project specification
    MIN_BLUR = 20.0
    MIN_ILLUMINATION = 20.0
    MAX_ILLUMINATION = 240.0

    blur_pass = blur_variance >= MIN_BLUR
    illumination_pass = (
        MIN_ILLUMINATION <= mean_illumination <= MAX_ILLUMINATION
    )

    gradable = blur_pass and illumination_pass

    # Explain failure
    reasons = []

    if not blur_pass:
        reasons.append(
            "Image is too blurry for reliable screening."
        )

    if not illumination_pass:
        reasons.append(
            "Image illumination is outside the acceptable range."
        )

    if gradable:
        message = "Image is suitable for screening."
    else:
        message = "Image is not suitable for reliable screening."

    return {
        "filename": file.filename,
        "gradable": gradable,
        "blur_variance": round(blur_variance, 4),
        "mean_illumination": round(mean_illumination, 4),
        "blur_threshold": MIN_BLUR,
        "illumination_range": [
            MIN_ILLUMINATION,
            MAX_ILLUMINATION
        ],
        "blur_pass": blur_pass,
        "illumination_pass": illumination_pass,
        "message": message,
        "reasons": reasons
    }
@app.post("/auth/register")
def register(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):
    try:
        existing_user = db.query(User).filter(
            User.email == user_data.email
        ).first()

        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered."
            )

        hashed_password = pwd_context.hash(
            user_data.password
        )

        new_user = User(
            name=user_data.name,
            email=user_data.email,
            password_hash=hashed_password
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "User registered successfully.",
            "user_id": new_user.id,
            "name": new_user.name,
            "email": new_user.email
        }

    except HTTPException:
        raise

    except Exception as e:
        print("REGISTER ERROR:", repr(e))
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Registration error: {str(e)}"
        )
    # Check whether email already exists
    existing_user = db.query(User).filter(
        User.email == user_data.email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered."
        )

    # Hash the password
    hashed_password = pwd_context.hash(
        user_data.password
    )

    # Create new user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully.",
        "user_id": new_user.id,
        "name": new_user.name,
        "email": new_user.email
    }
@app.post("/auth/login")
def login(
    user_data: LoginRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.email == user_data.email
    ).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    password_valid = pwd_context.verify(
        user_data.password,
        user.password_hash
    )

    if not password_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email
        }
    )

    return {
        "message": "Login successful.",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "name": user.name,
        "email": user.email
    }
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    start_time = time.perf_counter()
    image_bytes = await file.read()

    try:
        img_tensor = preprocess_fundus(image_bytes, target_size=384)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Please upload a clear JPEG or PNG fundus scan."
        )

    # 1. Forward Pass through ONNX Runtime
    outputs = session.run(None, {input_name: img_tensor})
    raw_continuous_score = float(outputs[0].flatten()[0])

    # 2. Dynamic Threshold Discretization
    predicted_grade = int(np.clip(np.digitize(raw_continuous_score, calibrated_thresholds), 0, 4))
    stage_name, referable, urgency, action = CLINICAL_STAGES[predicted_grade]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    prediction = Prediction(
    user_id=current_user.id,
        filename=file.filename,
        eye=None,
        predicted_grade=predicted_grade,
        stage_name=stage_name,
        continuous_score=raw_continuous_score,
        referable=referable,
        urgency=urgency,
        clinical_action=action
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "filename": file.filename,
        "predicted_grade": predicted_grade,
        "stage_name": stage_name,
        "continuous_score": round(raw_continuous_score, 4),
        "referable": referable,
        "urgency": urgency,
        "clinical_action": action,
        "thresholds_applied": calibrated_thresholds,
        "latency_ms": latency_ms
    }

@app.post("/screen/batch")
async def batch_predict(
    od: UploadFile = File(...),
    os: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Predict diabetic retinopathy for both eyes.

    OD = Right eye
    OS = Left eye

    Final grade is based on the worse eye.
    """

    async def predict_eye(file: UploadFile):
        start_time = time.perf_counter()

        image_bytes = await file.read()

        try:
            img_tensor = preprocess_fundus(
                image_bytes,
                target_size=384
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format for {file.filename}."
            )

        # Run ONNX model
        outputs = session.run(
            None,
            {input_name: img_tensor}
        )

        raw_continuous_score = float(
            outputs[0].flatten()[0]
        )

        # Convert continuous score to DR grade 0–4
        predicted_grade = int(
            np.clip(
                np.digitize(
                    raw_continuous_score,
                    calibrated_thresholds
                ),
                0,
                4
            )
        )

        stage_name, referable, urgency, action = CLINICAL_STAGES[
            predicted_grade
        ]

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        return {
            "filename": file.filename,
            "predicted_grade": predicted_grade,
            "stage_name": stage_name,
            "continuous_score": round(
                raw_continuous_score,
                4
            ),
            "referable": referable,
            "urgency": urgency,
            "clinical_action": action,
            "latency_ms": latency_ms
        }
    screening_id = str(uuid.uuid4())
    # Predict right eye (OD)
    od_result = await predict_eye(od)

    # Predict left eye (OS)
    os_result = await predict_eye(os)
    # Save right eye (OD) prediction
    od_prediction = Prediction(
    user_id=current_user.id,
    screening_id=screening_id,
        filename=od_result["filename"],
        eye="OD",
        predicted_grade=od_result["predicted_grade"],
        stage_name=od_result["stage_name"],
        continuous_score=od_result["continuous_score"],
        referable=od_result["referable"],
        urgency=od_result["urgency"],
        clinical_action=od_result["clinical_action"]
    )

    # Save left eye (OS) prediction
    os_prediction = Prediction(
    user_id=current_user.id,
    screening_id=screening_id,
        filename=os_result["filename"],
        eye="OS",
        predicted_grade=os_result["predicted_grade"],
        stage_name=os_result["stage_name"],
        continuous_score=os_result["continuous_score"],
        referable=os_result["referable"],
        urgency=os_result["urgency"],
        clinical_action=os_result["clinical_action"]
    )

    db.add(od_prediction)
    db.add(os_prediction)
    db.commit()
    # Worse-eye rule
    final_grade = max(
        od_result["predicted_grade"],
        os_result["predicted_grade"]
    )

    final_stage, final_referable, final_urgency, final_action = (
        CLINICAL_STAGES[final_grade]
    )

    return {
        "screening_id": screening_id,
        "right_eye_OD": od_result,
        "left_eye_OS": os_result,
        "final_result": {
            "predicted_grade": final_grade,
            "stage_name": final_stage,
            "referable": final_referable,
            "urgency": final_urgency,
            "clinical_action": final_action
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

@app.get("/predictions/history")
def prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).all()

    return {
        "user_id": current_user.id,
        "total_predictions": len(predictions),
        "predictions": [
            {
                "id": prediction.id,
                "filename": prediction.filename,
                "eye": prediction.eye,
                "predicted_grade": prediction.predicted_grade,
                "stage_name": prediction.stage_name,
                "continuous_score": prediction.continuous_score,
                "referable": prediction.referable,
                "urgency": prediction.urgency,
                "clinical_action": prediction.clinical_action,
                "created_at": prediction.created_at
            }
            for prediction in predictions
        ]
    }