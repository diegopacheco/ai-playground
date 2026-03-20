import { useState } from "react";

interface UserInputProps {
  onAnalyze: (input: string) => void;
  isLoading: boolean;
}

function UserInput({ onAnalyze, isLoading }: UserInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onAnalyze(value);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="https://github.com/username or username"
        className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
      />
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium px-6 py-2 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
      >
        {isLoading ? "Analyzing..." : "Analyze"}
      </button>
    </form>
  );
}

export default UserInput;
