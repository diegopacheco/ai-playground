import { useState } from "react";

interface UserInputProps {
  onAnalyze: (input: string) => void;
  isLoading: boolean;
}

function UserInput({ onAnalyze, isLoading }: UserInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onAnalyze(value);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={isLoading}
          placeholder="https://github.com/username or username"
          className="flex-1 bg-gray-50 border border-gray-300 rounded-lg px-4 py-2 text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !value.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-300 disabled:text-gray-500 text-white font-medium px-6 py-2 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
        >
          {isLoading ? "Analyzing..." : "Analyze"}
        </button>
      </form>
      {isLoading && (
        <div className="flex items-center gap-3 px-2">
          <span className="inline-block w-5 h-5 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-blue-600 font-semibold text-lg animate-pulse">
            Searching the truth...
          </span>
        </div>
      )}
    </div>
  );
}

export default UserInput;
