'use client';

import { useEffect, useRef, useState } from 'react';

interface VideoPlayerProps {
  videoFile: File;
  mode: 'manual' | 'automatic';
  exerciseId: number;
  onStatsUpdate: (stats: {
    count: number;
    stage: string;
    angle: number | null;
    exerciseName: string;
  }) => void;
  onReset: () => void;
}

interface PredictionResult {
  exerciseId: number;
  exerciseName: string;
  confidence: number;
  totalPredictions: number;
  isFinal: boolean;
}

export default function VideoPlayer({
  videoFile,
  mode,
  exerciseId,
  onStatsUpdate,
  onReset,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(30);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const isWaitingForResponse = useRef(false);
  const isPausedRef = useRef(false);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement) return;

    // Load video file
    const videoUrl = URL.createObjectURL(videoFile);
    videoElement.src = videoUrl;

    return () => {
      URL.revokeObjectURL(videoUrl);
    };
  }, [videoFile]);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/process');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsProcessing(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.success) {
        // Draw processed frame on canvas
        const canvas = canvasRef.current;
        if (canvas && data.frame) {
          const ctx = canvas.getContext('2d', { alpha: false });
          if (!ctx) return;

          const img = new Image();
          img.onload = () => {
            if (!canvas || !ctx) return;

            // 使用 requestAnimationFrame 來確保平滑繪製
            requestAnimationFrame(() => {
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

              // 如果有警告，在畫面上顯示
              if (data.warning) {
                ctx.fillStyle = 'rgba(255, 165, 0, 0.8)';
                ctx.fillRect(10, 10, 300, 40);
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.fillText(data.warning, 20, 35);
              }
            });
          };
          img.src = data.frame;
        }

        // 只在未暫停時更新統計數據
        if (!isPausedRef.current) {
          // Update prediction result (如果有預測資料)
          if (data.predicted_exercise_id !== undefined) {
            setPrediction({
              exerciseId: data.predicted_exercise_id,
              exerciseName: data.predicted_exercise_name,
              confidence: data.prediction_confidence,
              totalPredictions: data.total_predictions,
              isFinal: data.is_prediction_final || false,
            });
          }

          // Update stats (使用預測的運動名稱，如果有的話)
          onStatsUpdate({
            count: data.count,
            stage: data.stage,
            angle: data.angle,
            exerciseName: data.predicted_exercise_name || data.exercise_name,
          });
        }
      } else {
        console.error('Frame processing error:', data.error);
        setError(data.error);
      }

      // 收到回應後，標記為可以發送下一幀
      isWaitingForResponse.current = false;
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Make sure backend is running on port 8000.');
      setIsProcessing(false);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setIsProcessing(false);
    };

    wsRef.current = ws;
  };

  const startProcessing = () => {
    const videoElement = videoRef.current;
    const canvas = canvasRef.current;

    if (!videoElement || !canvas) return;

    // Set canvas size to match video
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    // Connect WebSocket
    connectWebSocket();

    // Start video playback
    videoElement.play();

    // Send frames at regular intervals (只在沒有等待回應且未暫停時發送)
    const frameInterval = 1000 / fps;
    frameIntervalRef.current = setInterval(() => {
      if (videoElement.ended) {
        stopProcessing();
        return;
      }

      // 只有在沒有等待回應且未暫停時才發送新幀
      if (!isWaitingForResponse.current && !isPausedRef.current) {
        sendFrame();
      }
    }, frameInterval);
  };

  const sendFrame = () => {
    const videoElement = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;

    if (!videoElement || !canvas || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    // 標記正在等待回應
    isWaitingForResponse.current = true;

    // 創建臨時 canvas 來捕獲當前影片幀（不影響顯示的 canvas）
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    if (!tempCtx) return;

    // 在臨時 canvas 上繪製當前影片幀
    tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);

    // Convert temp canvas to base64
    const frameBase64 = tempCanvas.toDataURL('image/jpeg', 0.8);

    // Send to backend
    ws.send(
      JSON.stringify({
        mode,
        exercise_id: exerciseId,
        frame: frameBase64,
        debug: false,
      })
    );
  };

  const pauseProcessing = () => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    isPausedRef.current = true;
    setIsPaused(true);
  };

  const resumeProcessing = () => {
    const videoElement = videoRef.current;
    if (videoElement) {
      videoElement.play();
    }
    isPausedRef.current = false;
    setIsPaused(false);
  };

  const stopProcessing = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
    }

    isPausedRef.current = false;
    setIsProcessing(false);
    setIsPaused(false);
  };

  const handleVideoLoaded = () => {
    const videoElement = videoRef.current;
    if (videoElement) {
      // Estimate FPS (default to 30 if not available)
      setFps(30);
    }
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 space-y-4">
      {/* Video Display */}
      <div className="relative aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          onLoadedMetadata={handleVideoLoaded}
          className="hidden"
          muted
        >
          {/* Captions track removed to avoid empty src warning.
              Add back with valid VTT file path if needed. */}
        </video>
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full object-contain"
        />
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 text-red-200">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-3">
        {!isProcessing ? (
          <button
            type='button'
            onClick={startProcessing}
            className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
          >
            開始處理
          </button>
        ) : isPaused ? (
          <button
            type='button'
            onClick={resumeProcessing}
            className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
          >
            繼續
          </button>
        ) : (
          <button
            type='button'
            onClick={pauseProcessing}
            className="flex-1 px-6 py-3 bg-yellow-600 hover:bg-yellow-700 text-white font-semibold rounded-lg transition-colors"
          >
            暫停
          </button>
        )}

        <button
          type='button'
          onClick={onReset}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors"
        >
          重設
        </button>
      </div>

      {/* Prediction Result Display */}
      {prediction && (
        <div className="bg-gradient-to-r bg-gray-800 border border-bityo/50 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-blue-200">
              {prediction.isFinal ? '最終預測' : '即時預測'}
            </h3>
            <span className="text-xs text-gray-400">
            </span>
          </div>

          <div className="space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white">
                運動名稱：{prediction.exerciseName}
              </span>
              <span className={`text-xl font-semibold ${prediction.confidence > 0.8 ? 'text-bityo' : prediction.confidence > 0.5 ? 'text-yellow-500' : 'text-red-500'}`}>
                準確度：{(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            {/* Confidence Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
              <div
                className={`bg-gradient-to-r h-2.5 rounded-full transition-all duration-300 ${prediction.confidence > 0.8 ? 'bg-bityo' : prediction.confidence > 0.5 ? 'bg-yellow-400' : 'bg-red-500'}`}
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
          </div>

          {prediction.isFinal && (
            <div className="text-sm text-green-300 text-center pt-2 border-t border-blue-500/30">
              分析完成
            </div>
          )}
        </div>
      )}
    </div>
  );
}
