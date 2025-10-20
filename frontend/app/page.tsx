'use client';

import { useState } from 'react';
import VideoUploader from '@/components/VideoUploader';
import VideoPlayer from '@/components/VideoPlayer';
import ExerciseSelector from '@/components/ExerciseSelector';
import StatsPanel from '@/components/StatsPanel';

export default function Home() {
  const [mode, setMode] = useState<'manual' | 'automatic'>('automatic');
  const [selectedExercise, setSelectedExercise] = useState<number>(3);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [stats, setStats] = useState({
    count: 0,
    stage: 'down',
    angle: null as number | null,
    exerciseName: '',
  });

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">
            💪 Fitness AI Trainer
          </h1>
          <p className="text-gray-300 text-lg">
            AI-powered exercise recognition and counting
          </p>
        </header>

        {/* Mode Selection */}
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={() => setMode('automatic')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              mode === 'automatic'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
                : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
          >
            🤖 Automatic Mode
          </button>
          <button
            onClick={() => setMode('manual')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              mode === 'manual'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
                : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
          >
            ✋ Manual Mode
          </button>
        </div>

        {/* Manual Mode: Exercise Selector */}
        {mode === 'manual' && (
          <div className="mb-8">
            <ExerciseSelector
              selectedExercise={selectedExercise}
              onSelect={setSelectedExercise}
            />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Video Upload */}
          <div className="lg:col-span-2">
            {!videoFile ? (
              <VideoUploader onFileSelect={setVideoFile} />
            ) : (
              <VideoPlayer
                videoFile={videoFile}
                mode={mode}
                exerciseId={selectedExercise}
                onStatsUpdate={setStats}
                onReset={() => {
                  setVideoFile(null);
                  setStats({ count: 0, stage: 'down', angle: null, exerciseName: '' });
                }}
              />
            )}
          </div>

          {/* Right: Stats Panel */}
          <div>
            <StatsPanel
              count={stats.count}
              stage={stats.stage}
              angle={stats.angle}
              exerciseName={stats.exerciseName}
              mode={mode}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-400 text-sm">
          <p>Powered by BiLSTM + MediaPipe | Built with Next.js + FastAPI</p>
        </footer>
      </div>
    </main>
  );
}
