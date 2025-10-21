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
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(30);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

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
        if (canvas) {
          const ctx = canvas.getContext('2d');
          const img = new Image();
          img.onload = () => {
            ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
          };
          img.src = data.frame;
        }

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
      } else {
        console.error('Frame processing error:', data.error);
      }
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

    // Send frames at regular intervals
    const frameInterval = 1000 / fps;
    frameIntervalRef.current = setInterval(() => {
      if (videoElement.paused || videoElement.ended) {
        stopProcessing();
        return;
      }

      sendFrame();
    }, frameInterval);
  };

  const sendFrame = () => {
    const videoElement = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;

    if (!videoElement || !canvas || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    // Draw current video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx?.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64
    const frameBase64 = canvas.toDataURL('image/jpeg', 0.8);

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

    setIsProcessing(false);
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
      <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
        <video
          ref={videoRef}
          onLoadedMetadata={handleVideoLoaded}
          className={`absolute inset-0 w-full h-full ${isProcessing ? 'opacity-0' : 'opacity-100'}`}
          controls={!isProcessing}
        >
          {/* Provide a captions track to satisfy accessibility/lint rules.
              If you have an actual VTT file, set src="/path/to/captions.vtt" and adjust srcLang/label accordingly. */}
          <track kind="captions" src="" srcLang="en" label="English" default />
        </video>
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full ${isProcessing ? 'opacity-100' : 'opacity-0'}`}
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
            ▶️ Start Processing
          </button>
        ) : (
          <button
            type='button'
            onClick={stopProcessing}
            className="flex-1 px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors"
          >
            ⏸️ Stop
          </button>
        )}

        <button
          type='button'
          onClick={onReset}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors"
        >
          🔄 Reset
        </button>
      </div>

      {/* Prediction Result Display */}
      {prediction && (
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/50 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-blue-200">
              {prediction.isFinal ? '🎯 Final Prediction' : '🔄 Real-time Prediction'}
            </h3>
            <span className="text-xs text-gray-400">
              {prediction.totalPredictions} predictions
            </span>
          </div>

          <div className="space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white">
                {prediction.exerciseName}
              </span>
              <span className="text-xl font-semibold text-green-400">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            {/* Confidence Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-green-500 to-blue-500 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
          </div>

          {prediction.isFinal && (
            <div className="text-sm text-green-300 text-center pt-2 border-t border-blue-500/30">
              ✓ Analysis Complete
            </div>
          )}
        </div>
      )}

      {/* Info */}
      <div className="text-sm text-gray-400 text-center">
        {isProcessing ? (
          <span className="text-green-400">● Processing video in real-time...</span>
        ) : (
          <span>Click Start Processing to begin analysis</span>
        )}
      </div>
    </div>
  );
}
