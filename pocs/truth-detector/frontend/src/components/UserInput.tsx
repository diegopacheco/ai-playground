import { useState, useEffect } from "react";
import type { AgentInfo } from "../api/client";

interface UserInputProps {
  onAnalyze: (input: string, cli: string, model: string) => void;
  isLoading: boolean;
  agents: AgentInfo[];
}

function UserInput({ onAnalyze, isLoading, agents }: UserInputProps) {
  const [value, setValue] = useState("");
  const [selectedCli, setSelectedCli] = useState("");
  const [selectedModel, setSelectedModel] = useState("");

  const currentAgent = agents.find((a) => a.name === selectedCli);
  const models = currentAgent?.models ?? [];

  useEffect(() => {
    if (agents.length > 0 && !selectedCli) {
      setSelectedCli(agents[0].name);
      setSelectedModel(agents[0].models[0]);
    }
  }, [agents, selectedCli]);

  useEffect(() => {
    if (models.length > 0 && !models.includes(selectedModel)) {
      setSelectedModel(models[0]);
    }
  }, [selectedCli, models, selectedModel]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim() && !isLoading && selectedCli && selectedModel) {
      onAnalyze(value, selectedCli, selectedModel);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="flex gap-3">
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
            disabled={isLoading || !value.trim() || !selectedCli || !selectedModel}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-300 disabled:text-gray-500 text-white font-medium px-6 py-2 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
          >
            {isLoading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
        <div className="flex gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">Agent:</label>
          <select
            value={selectedCli}
            onChange={(e) => setSelectedCli(e.target.value)}
            disabled={isLoading}
            className="bg-gray-50 border border-gray-300 rounded-lg px-3 py-1.5 text-gray-900 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          >
            {agents.map((a) => (
              <option key={a.name} value={a.name}>{a.name}</option>
            ))}
          </select>
          <label className="text-sm font-medium text-gray-700">Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isLoading}
            className="bg-gray-50 border border-gray-300 rounded-lg px-3 py-1.5 text-gray-900 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          >
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
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
