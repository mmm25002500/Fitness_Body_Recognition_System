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
      console.log('✅ WebSocket connected');
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

        // Update stats
        onStatsUpdate({
          count: data.count,
          stage: data.stage,
          angle: data.angle,
          exerciseName: data.exercise_name,
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
        />
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
            onClick={startProcessing}
            className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
          >
            ▶️ Start Processing
          </button>
        ) : (
          <button
            onClick={stopProcessing}
            className="flex-1 px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors"
          >
            ⏸️ Stop
          </button>
        )}

        <button
          onClick={onReset}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors"
        >
          🔄 Reset
        </button>
      </div>

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
