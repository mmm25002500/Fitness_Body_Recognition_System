'use client';

import { useCallback } from 'react';

interface VideoUploaderProps {
  onFileSelect: (file: File) => void;
}

export default function VideoUploader({ onFileSelect }: VideoUploaderProps) {
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="border-4 border-dashed border-purple-500 rounded-2xl p-12 text-center bg-slate-800/50 backdrop-blur-sm hover:bg-slate-800/70 transition-all cursor-pointer"
    >
      <div className="flex flex-col items-center gap-4">
        <div className="text-6xl">🎥</div>
        <h3 className="text-2xl font-semibold text-white">Upload Your Workout Video</h3>
        <p className="text-gray-400">Drag & drop or click to browse</p>

        <label className="mt-4 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg cursor-pointer transition-colors">
          Choose Video
          <input
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="hidden"
          />
        </label>

        <div className="mt-6 text-sm text-gray-500 space-y-2">
          <p>✅ Full body in frame</p>
          <p>✅ Good lighting</p>
          <p>✅ Clear movements</p>
        </div>
      </div>
    </div>
  );
}
