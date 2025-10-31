'use client';

interface StatsPanelProps {
  count: number;
  stage: string;
  angle: number | null;
  exerciseName: string;
  mode: 'manual' | 'automatic';
}

export default function StatsPanel({ count, stage, angle, exerciseName, mode }: StatsPanelProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 space-y-6">
      {/* Mode Badge */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">統計</h3>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${mode === 'automatic'
          ? 'bg-purple-600/20 text-purple-300'
          : 'bg-blue-600/20 text-blue-300'
          }`}>
          {mode === 'automatic' ? '自動' : '手動'}
        </span>
      </div>

      {/* Exercise Name */}
      {exerciseName && (
        <div className="bg-slate-700/50 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-1">運動</div>
          <div className="text-lg font-semibold text-white">{exerciseName}</div>
        </div>
      )}

      {/* Rep Count */}
      <div className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl p-6 text-center shadow-lg shadow-purple-500/30">
        <div className="text-sm text-purple-100 mb-2">次數</div>
        <div className="text-6xl font-bold text-white">{count}</div>
      </div>

      {/* Stage */}
      <div className="bg-slate-700/50 rounded-xl p-4">
        <div className="text-sm text-gray-400 mb-2">當前階段</div>
        <div className={`text-2xl font-bold ${stage?.toLowerCase() === 'up' ? 'text-green-400' : 'text-orange-400'
          }`}>
          {stage?.toUpperCase() === 'UP' ? '上升' : '下降'}
        </div>
      </div>

      {/* Angle */}
      {angle !== null && (
        <div className="bg-slate-700/50 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-2">關節角度</div>
          <div className="text-3xl font-bold text-white">{angle.toFixed(1)}°</div>

          {/* Visual angle indicator */}
          <div className="mt-3 h-2 bg-slate-600 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-orange-400 via-yellow-400 to-green-400 transition-all duration-300"
              style={{ width: `${Math.min((angle / 180) * 100, 100)}%` }}
            />
          </div>
        </div>
      )}

      <div className="bg-slate-700/30 rounded-xl p-4 text-sm text-gray-400">
        <div className="font-semibold text-gray-300 mb-2">提示</div>
        <ul className="space-y-1">
          <li>• 保持全身在畫面內</li>
          <li>• 完成完整的動作</li>
          <li>• 維持穩定的節奏</li>
        </ul>
      </div>
    </div>
  );
}
