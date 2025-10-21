#!/bin/bash
# 上傳影片並取得暫存路徑
echo "1. 上傳影片..."
UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8000/api/upload \
  -F "file=@/Users/tershi/Project/專題/example_video1.mp4")

echo "$UPLOAD_RESPONSE" | python3 -m json.tool

TEMP_PATH=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['temp_path'])")

echo ""
echo "2. 預測運動類型..."
echo "暫存路徑: $TEMP_PATH"

PREDICT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/predict \
  -F "video_path=$TEMP_PATH" \
  -F "window_size=45" \
  -F "stride=3")

echo "$PREDICT_RESPONSE" | python3 -m json.tool
