"""
FastAPI Backend for Fitness AI Trainer
Provides endpoints for video processing, pose estimation, and exercise counting
"""
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import os
from typing import Optional, Dict, Any
import json
import base64
import torch

from pose_extractor_mediapipe import PoseExtractorMediaPipe
from exercise_counter import RepetitionCounter
from model import BiLSTMAttention
from feature_utils_v2 import landmarks_to_features_v2

app = FastAPI(title="Fitness AI Trainer API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pose_extractor = None
model = None
device = None
exercise_names = [
    "Barbell Biceps Curl",
    "Hammer Curl",
    "Push-up",
    "Shoulder Press",
    "Squat"
]

@app.on_event("startup")
async def startup_event():
    """Initialize model and pose extractor on startup"""
    global pose_extractor, model, device

    # Initialize MediaPipe
    pose_extractor = PoseExtractorMediaPipe()

    # Load BiLSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttention(
        input_dim=102,
        hidden_dim=96,
        attn_dim=128,
        num_classes=5
    )

    model_path = os.path.join(os.path.dirname(__file__), "bilstm_mix_best_pt.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print("MediaPipe initialized")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "pose_extractor_ready": pose_extractor is not None
    }

@app.get("/api/exercises")
async def get_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            {"id": 0, "name": "Barbell Biceps Curl"},
            {"id": 1, "name": "Hammer Curl"},
            {"id": 2, "name": "Push-up"},
            {"id": 3, "name": "Shoulder Press"},
            {"id": 4, "name": "Squat"}
        ]
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video and return temporary file path"""
    try:
        # Save to temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Get video info
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "success": True,
            "video_id": os.path.basename(tmp_path),
            "temp_path": tmp_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/predict")
async def predict_exercise(
    video_path: str = Form(...),
    window_size: int = Form(45),
    stride: int = Form(3)
):
    """Predict exercise type from video"""
    try:
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Video file not found"}
            )

        # Extract features from video
        cap = cv2.VideoCapture(video_path)
        frame_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, _ = pose_extractor.extract_landmarks(frame)
            if landmarks is not None:
                features = landmarks_to_features_v2([landmarks])[0]
                frame_features.append(features)

        cap.release()

        if len(frame_features) < window_size:
            return {
                "success": False,
                "error": f"Video too short. Need at least {window_size} frames with pose detected"
            }

        # Sliding window prediction
        predictions = []

        for i in range(0, len(frame_features) - window_size + 1, stride):
            window = frame_features[i:i + window_size]
            window_tensor = torch.FloatTensor([window]).to(device)

            with torch.no_grad():
                output = model(window_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

                predictions.append({
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities[0].cpu().numpy().tolist()
                })

        # Get most common prediction
        class_counts = {}
        for pred in predictions:
            cls = pred["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        final_class = max(class_counts, key=class_counts.get)
        avg_confidence = np.mean([p["confidence"] for p in predictions if p["class"] == final_class])

        return {
            "success": True,
            "exercise_id": final_class,
            "exercise_name": exercise_names[final_class],
            "confidence": float(avg_confidence),
            "total_predictions": len(predictions),
            "class_distribution": class_counts
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing

    Receives: JSON with {mode, exercise_id, frame_base64, debug}
    Sends: JSON with {frame_base64, count, stage, angle, exercise_name}
    """
    await websocket.accept()

    # Create counter instance
    counter = None

    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_text()
            message = json.loads(data)

            mode = message.get("mode", "manual")  # "manual" or "automatic"
            exercise_id = message.get("exercise_id", 3)
            frame_base64 = message.get("frame")
            debug = message.get("debug", False)

            # Decode frame
            frame_bytes = base64.b64decode(frame_base64.split(',')[1] if ',' in frame_base64 else frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract pose
            landmarks, pose_landmarks = pose_extractor.extract_landmarks(frame)

            if landmarks is None:
                # No pose detected
                await websocket.send_json({
                    "success": False,
                    "error": "No pose detected"
                })
                continue

            # Initialize or update counter
            if counter is None or (mode == "manual" and counter.exercise_type != exercise_id):
                counter = RepetitionCounter(exercise_type=exercise_id)

            # Update counter with exercise_id
            result = counter.update(landmarks, exercise_id)
            count = result["count"]
            stage = result["stage"]
            angle = result["angle"]

            # Draw visualization
            from visualization import PoseVisualizer
            visualizer = PoseVisualizer()

            # Draw skeleton (need to get pose_landmarks again)
            _, pose_landmarks_obj = pose_extractor.extract_landmarks(frame)
            if pose_landmarks_obj is not None:
                frame = visualizer.draw_landmarks(frame, pose_landmarks_obj)

            # Draw counter info
            exercise_name = exercise_names[exercise_id]
            frame = visualizer.draw_counter_info(frame, exercise_name, count, stage)

            # Draw angle if available
            if angle is not None and result.get("points") is not None:
                _, point_b, _ = result["points"]
                frame = visualizer.draw_angle(frame, angle, point_b)

            # Encode frame back to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64_out = base64.b64encode(buffer).decode('utf-8')

            # Send response
            await websocket.send_json({
                "success": True,
                "frame": f"data:image/jpeg;base64,{frame_base64_out}",
                "count": count,
                "stage": stage,
                "angle": angle,
                "exercise_name": exercise_name,
                "exercise_id": exercise_id
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
