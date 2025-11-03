'use client';

import { useState } from 'react';
import VideoUploader from '@/components/VideoUploader';
import VideoPlayer from '@/components/VideoPlayer';
// import ExerciseSelector from '@/components/ExerciseSelector';
import StatsPanel from '@/components/StatsPanel';
import Footer from '@/components/Footer/Footer';

export default function HomeClient() {
  const mode = 'automatic';
  const selectedExercise = 3;
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [stats, setStats] = useState({
    count: 0,
    stage: 'down',
    angle: null as number | null,
    exerciseName: '',
  });

  return (
    <>
      {/* <div className="flex justify-center gap-4 mb-8">
        <button
          type='button'
          onClick={() => setMode('automatic')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'automatic'
            ? 'bg-neutral-black text-white'
            : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
            }`}
        >
          自動模式
        </button>
        <button
          type='button'
          onClick={() => setMode('manual')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'manual'
            ? 'bg-neutral-black text-white'
            : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
            }`}
        >
          手動模式
        </button>
      </div> */}

      {/* {mode === 'manual' && (
        <div className="mb-8">
          <ExerciseSelector
            selectedExercise={selectedExercise}
            onSelect={setSelectedExercise}
          />
        </div>
      )} */}

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
