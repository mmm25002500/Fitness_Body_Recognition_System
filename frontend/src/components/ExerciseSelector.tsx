'use client';

interface ExerciseSelectorProps {
  selectedExercise: number;
  onSelect: (exerciseId: number) => void;
}

const exercises = [
  { id: 0, name: '槓鈴二頭肌彎舉' },
  { id: 1, name: '錘式彎舉' },
  { id: 2, name: '伏地挺身' },
  { id: 3, name: '肩上推舉' },
  { id: 4, name: '深蹲' },
];

export default function ExerciseSelector({ selectedExercise, onSelect }: ExerciseSelectorProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6">
      <h3 className="text-xl font-semibold text-white mb-4">選擇運動類型</h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {exercises.map((exercise) => (
          <button
            type="button"
            key={exercise.id}
            onClick={() => onSelect(exercise.id)}
            className={`p-4 rounded-xl transition-all ${selectedExercise === exercise.id
              ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50 scale-105'
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
          >
            <div className="text-sm font-medium">{exercise.name}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
