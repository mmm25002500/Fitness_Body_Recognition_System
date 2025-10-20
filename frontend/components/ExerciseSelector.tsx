'use client';

interface ExerciseSelectorProps {
  selectedExercise: number;
  onSelect: (exerciseId: number) => void;
}

const exercises = [
  { id: 0, name: 'Barbell Biceps Curl', emoji: '💪' },
  { id: 1, name: 'Hammer Curl', emoji: '🔨' },
  { id: 2, name: 'Push-up', emoji: '🤸' },
  { id: 3, name: 'Shoulder Press', emoji: '🏋️' },
  { id: 4, name: 'Squat', emoji: '🦵' },
];

export default function ExerciseSelector({ selectedExercise, onSelect }: ExerciseSelectorProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6">
      <h3 className="text-xl font-semibold text-white mb-4">Select Exercise Type</h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {exercises.map((exercise) => (
          <button
            key={exercise.id}
            onClick={() => onSelect(exercise.id)}
            className={`p-4 rounded-xl transition-all ${
              selectedExercise === exercise.id
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/50 scale-105'
                : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
          >
            <div className="text-3xl mb-2">{exercise.emoji}</div>
            <div className="text-sm font-medium">{exercise.name}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
