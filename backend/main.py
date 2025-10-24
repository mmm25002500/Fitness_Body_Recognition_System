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
from model import BiLSTMAttention, BiLSTMSimple, BiLSTMSingleLayer
from feature_utils_v2 import landmarks_to_features_v2, landmarks_to_features_simple

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
# 使用原本的模型 (5 classes - 包含深蹲)
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

    # Load BiLSTM model (使用 bilstm_mix_best_pt.pth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(os.path.dirname(__file__), "bilstm_mix_best_pt.pth")

    print("載入模型 bilstm_mix_best_pt.pth (支援 5 種運動)...")
    model = BiLSTMAttention(
        input_dim=102,
        hidden_dim=96,
        attn_dim=128,
        num_classes=5
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"✓ 模型載入成功 (input_dim=102, num_classes=5)")
    print(f"✓ 支援運動: {', '.join(exercise_names)}")
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
                # 使用 102 維特徵提取（對應 bilstm_mix_best_pt.pth）
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
    WebSocket endpoint for real-time video processing with automatic exercise prediction

    Receives: JSON with {mode, exercise_id, frame_base64, debug, is_video_end}
    Sends: JSON with {frame_base64, count, stage, angle, exercise_name, predicted_exercise_id, prediction_confidence}
    """
    await websocket.accept()

    # Create counter instance
    counter = None

    # Prediction state (累積預測)
    frame_features = []  # 累積特徵向量
    predictions_history = []  # 所有預測記錄
    window_size = 45
    stride = 3
    predicted_exercise_id = None
    prediction_confidence = 0.0

    # Smoothing filter for angle and other values
    from collections import deque
    angle_history = deque(maxlen=5)  # 保留最近5幀的角度
    count_history = deque(maxlen=3)  # 保留最近3幀的計數
    stage_history = deque(maxlen=3)  # 保留最近3幀的階段

    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_text()
            message = json.loads(data)

            mode = message.get("mode", "manual")  # "manual" or "automatic"
            exercise_id = message.get("exercise_id", 3)
            frame_base64 = message.get("frame")
            debug = message.get("debug", False)
            is_video_end = message.get("is_video_end", False)

            # Decode frame
            frame_bytes = base64.b64decode(frame_base64.split(',')[1] if ',' in frame_base64 else frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract pose
            landmarks, pose_landmarks = pose_extractor.extract_landmarks(frame)

            if landmarks is None:
                # No pose detected - 仍然回傳畫面，但標註警告
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64_out = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "success": True,
                    "frame": f"data:image/jpeg;base64,{frame_base64_out}",
                    "count": counter.count if counter else 0,
                    "stage": counter.stage if counter else "unknown",
                    "angle": None,
                    "exercise_name": exercise_names[exercise_id] if exercise_id < len(exercise_names) else "Unknown",
                    "exercise_id": exercise_id,
                    "warning": "No pose detected in this frame"
                })
                continue

            # 累積特徵並進行預測
            if mode == "automatic":
                # 提取 102 維特徵
                features = landmarks_to_features_v2([landmarks])[0]
                frame_features.append(features)

                # 當累積足夠幀數，開始滑動窗口預測
                if len(frame_features) >= window_size:
                    # 取最新的窗口
                    window = frame_features[-window_size:]
                    window_tensor = torch.FloatTensor([window]).to(device)

                    with torch.no_grad():
                        output = model(window_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        pred_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][pred_class].item()

                    predictions_history.append({
                        "class": pred_class,
                        "confidence": confidence
                    })

                    # 計算多數投票的預測結果
                    class_counts = {}
                    for pred in predictions_history:
                        cls = pred["class"]
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                    # 最多的類別
                    predicted_exercise_id = max(class_counts, key=class_counts.get)

                    # 計算該類別的平均信心度
                    same_class_preds = [p["confidence"] for p in predictions_history if p["class"] == predicted_exercise_id]
                    prediction_confidence = np.mean(same_class_preds) if same_class_preds else 0.0

                    # 自動模式下，使用預測的運動類型
                    exercise_id = predicted_exercise_id

            # Initialize or update counter
            if counter is None or (mode == "manual" and counter.exercise_type != exercise_id):
                counter = RepetitionCounter(exercise_type=exercise_id)

            # Update counter with exercise_id
            result = counter.update(landmarks, exercise_id)
            raw_count = result["count"]
            raw_stage = result["stage"]
            raw_angle = result["angle"]

            # Apply smoothing filter
            count_history.append(raw_count)
            stage_history.append(raw_stage)
            if raw_angle is not None:
                angle_history.append(raw_angle)

            # 使用最常出現的值（眾數）來減少閃爍
            from statistics import mode as stats_mode
            try:
                count = stats_mode(count_history) if len(count_history) > 0 else raw_count
                stage = stats_mode(stage_history) if len(stage_history) > 0 else raw_stage
            except:
                count = raw_count
                stage = raw_stage

            # 角度使用平均值來平滑
            if len(angle_history) > 0:
                angle = sum(angle_history) / len(angle_history)
            else:
                angle = raw_angle

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

            # Encode frame back to base64 with higher quality to reduce artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_base64_out = base64.b64encode(buffer).decode('utf-8')

            # Send response
            response = {
                "success": True,
                "frame": f"data:image/jpeg;base64,{frame_base64_out}",
                "count": count,
                "stage": stage,
                "angle": angle,
                "exercise_name": exercise_name,
                "exercise_id": exercise_id
            }

            # 加入預測資訊（如果有預測）
            if predicted_exercise_id is not None:
                response["predicted_exercise_id"] = predicted_exercise_id
                response["predicted_exercise_name"] = exercise_names[predicted_exercise_id]
                response["prediction_confidence"] = float(prediction_confidence)
                response["total_predictions"] = len(predictions_history)
                response["is_prediction_final"] = is_video_end

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
