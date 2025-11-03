'use client';

import { useCallback } from 'react';

interface VideoUploaderProps {
  onFileSelect: (file: File) => void;
}

export default function VideoUploader({ onFileSelect }: VideoUploaderProps) {
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file?.type?.startsWith('video/')) {
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
      className="border-[1px] border-dashed border-neutral-200 rounded-3xl p-12 text-center bg-slate-800/50 backdrop-blur-sm hover:bg-slate-800/90 transition-all cursor-pointer"
    >
      <div className="flex flex-col items-center gap-4">
        <h3 className="text-2xl font-semibold text-white">上傳您的運動影片</h3>
        <p className="text-gray-400">拖放或點擊以瀏覽</p>

        <label className="mt-4 px-6 py-3 bg-bityo/50 hover:bg-bityo/20 text-white font-semibold rounded-lg cursor-pointer transition-colors">
          選擇影片
          <input
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}
