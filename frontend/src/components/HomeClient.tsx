'use client';

import { useState } from 'react';
import VideoUploader from '@/components/VideoUploader';
import VideoPlayer from '@/components/VideoPlayer';
import ExerciseSelector from '@/components/ExerciseSelector';
import StatsPanel from '@/components/StatsPanel';
import Footer from '@/components/Footer/Footer';

export default function HomeClient() {
  const [mode, setMode] = useState<'manual' | 'automatic'>('automatic');
  const [selectedExercise, setSelectedExercise] = useState<number>(3);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [stats, setStats] = useState({
    count: 0,
    stage: 'down',
    angle: null as number | null,
    exerciseName: '',
  });

  return (
    <>
      <header className="text-center mb-8">
        <h1 className="text-5xl font-bold text-white mb-2">
          健身肢體辨識系統
        </h1>
        <p className="text-gray-300 text-lg">
          一個基於 BiLSTM 和 MediaPipe 的健身肢體辨識系統，可支援全自動或全手動識別五種不同的運動模式。
        </p>
      </header>

      <div className="flex justify-center gap-4 mb-8">
        <button
          type='button'
          onClick={() => setMode('automatic')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'automatic'
            ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
            : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
        >
          自動模式
        </button>
        <button
          type='button'
          onClick={() => setMode('manual')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'manual'
            ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50'
            : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
        >
          手動模式
        </button>
      </div>

      {mode === 'manual' && (
        <div className="mb-8">
          <ExerciseSelector
            selectedExercise={selectedExercise}
            onSelect={setSelectedExercise}
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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

      <Footer />
    </>
  );
}
