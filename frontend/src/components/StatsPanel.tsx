'use client';

interface StatsPanelProps {
  count: number;
  stage: string;
  angle: number | null;
  exerciseName: string;
  mode: 'manual' | 'automatic';
}

// 根據運動類型和角度判斷警示狀態
const getAngleStatus = (exerciseName: string, angle: number | null) => {
  if (angle === null) return null;

  const lowerExerciseName = exerciseName.toLowerCase();

  // 深蹲: <90 注意, <50 報警
  if (lowerExerciseName.includes('squat')) {
    if (angle < 50) return { level: 'danger', text: '報警' };
    if (angle < 90) return { level: 'warning', text: '注意' };
    return { level: 'normal', text: '正常' };
  }

  // 肩推/肩上推舉: <90 注意, <75 報警
  if (lowerExerciseName.includes('shoulder') && lowerExerciseName.includes('press')) {
    if (angle < 75) return { level: 'danger', text: '報警' };
    if (angle < 90) return { level: 'warning', text: '注意' };
    return { level: 'normal', text: '正常' };
  }

  // 槓鈴二頭彎舉: <140 注意
  if (lowerExerciseName.includes('bicep') || lowerExerciseName.includes('curl')) {
    if (angle < 140) return { level: 'warning', text: '注意' };
    return { level: 'normal', text: '正常' };
  }

  return { level: 'normal', text: '正常' };
};

export default function StatsPanel({ count, stage, angle, exerciseName, mode }: StatsPanelProps) {
  const angleStatus = getAngleStatus(exerciseName, angle);
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 space-y-6">
      {/* Mode Badge */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">統計</h3>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${mode === 'automatic'
          ? 'bg-neutral-600/20 text-bityo'
          : 'bg-blue-600/20 text-red-500'
          }`}>
          {mode === 'automatic' ? '自動模式' : '手動模式'}
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
      <div className="bg-slate-700/50 rounded-xl p-6 text-center">
        <div className="text-sm text-neutral-100 mb-2">次數</div>
        <div className={`text-6xl font-bold text-white ${count > 0 ? 'text-bityo' : 'text-white'}`}>{count}</div>
      </div>

      {/* Stage */}
      <div className="bg-slate-700/50 rounded-xl p-4">
        <div className="text-sm text-gray-400 mb-2">當前階段</div>
        <div className={`text-2xl font-bold ${stage?.toLowerCase() === 'up' ? 'text-bityo' : 'text-cyan-400'}`}>
          {stage?.toUpperCase() === 'UP' ? '上升' : '下降'}
        </div>
      </div>

      {/* Angle */}
      {angle !== null && (
        <div className="bg-slate-700/50 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-2">關節角度</div>
          <div className="flex items-baseline gap-3">
            <div className="text-3xl font-bold text-white">{angle.toFixed(1)}°</div>
            <span className={`text-lg font-medium ${angleStatus?.level === 'danger' ? 'text-func-error' :
              angleStatus?.level === 'warning' ? 'text-func-warning' :
                'text-func-success'
              }`}>
              {angleStatus?.text}
            </span>
          </div>

          {/* Visual angle indicator */}
          <div className="mt-3 h-2 bg-slate-600 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${angleStatus?.level === 'danger' ? 'bg-func-error' :
                angleStatus?.level === 'warning' ? 'bg-func-warning' :
                  angleStatus?.level === 'normal' ? 'bg-func-success' :
                    'bg-red-500'
                }`}
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
