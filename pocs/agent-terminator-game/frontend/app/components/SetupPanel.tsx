import { useState } from "react";
import { AVAILABLE_AGENTS, AGENT_COLORS } from "~/types";

interface Props {
  onStart: (config: {
    terminator_agent: string;
    terminator_model: string;
    mosquito_agent: string;
    mosquito_model: string;
    grid_size: string;
  }) => void;
  loading: boolean;
}

export default function SetupPanel({ onStart, loading }: Props) {
  const [terminatorAgent, setTerminatorAgent] = useState("claude");
  const [terminatorModel, setTerminatorModel] = useState("sonnet");
  const [mosquitoAgent, setMosquitoAgent] = useState("gemini");
  const [mosquitoModel, setMosquitoModel] = useState("gemini-3-flash");

  const getModels = (agentName: string) =>
    AVAILABLE_AGENTS.find((a) => a.name === agentName)?.models || [];

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-4xl font-bold text-center mb-2 text-white">
        TERMINATOR MOSQUITO GAME
      </h1>
      <p className="text-center text-gray-400 mb-10 text-lg">
        Choose your agents and let the battle begin
      </p>

      <div className="grid grid-cols-2 gap-8 mb-10">
        <div className="bg-[#0d2818] border border-red-900/50 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-3xl">🤖</span> Terminator
          </h2>
          <label className="block text-sm text-gray-400 mb-1">Agent</label>
          <select
            value={terminatorAgent}
            onChange={(e) => {
              setTerminatorAgent(e.target.value);
              setTerminatorModel(getModels(e.target.value)[0] || "");
            }}
            className="w-full bg-[#0f2e1a] border border-gray-700 rounded-lg p-3 mb-4 text-white"
          >
            {AVAILABLE_AGENTS.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
          <label className="block text-sm text-gray-400 mb-1">Model</label>
          <select
            value={terminatorModel}
            onChange={(e) => setTerminatorModel(e.target.value)}
            className="w-full bg-[#0f2e1a] border border-gray-700 rounded-lg p-3 text-white"
          >
            {getModels(terminatorAgent).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <div
            className="mt-4 h-1 rounded"
            style={{ backgroundColor: AGENT_COLORS[terminatorAgent] }}
          />
        </div>

        <div className="bg-[#0d2818] border border-green-900/50 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-3xl">🦟</span> Mosquitos
          </h2>
          <label className="block text-sm text-gray-400 mb-1">Agent</label>
          <select
            value={mosquitoAgent}
            onChange={(e) => {
              setMosquitoAgent(e.target.value);
              setMosquitoModel(getModels(e.target.value)[0] || "");
            }}
            className="w-full bg-[#0f2e1a] border border-gray-700 rounded-lg p-3 mb-4 text-white"
          >
            {AVAILABLE_AGENTS.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
          <label className="block text-sm text-gray-400 mb-1">Model</label>
          <select
            value={mosquitoModel}
            onChange={(e) => setMosquitoModel(e.target.value)}
            className="w-full bg-[#0f2e1a] border border-gray-700 rounded-lg p-3 text-white"
          >
            {getModels(mosquitoAgent).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <div
            className="mt-4 h-1 rounded"
            style={{ backgroundColor: AGENT_COLORS[mosquitoAgent] }}
          />
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={() =>
            onStart({
              terminator_agent: terminatorAgent,
              terminator_model: terminatorModel,
              mosquito_agent: mosquitoAgent,
              mosquito_model: mosquitoModel,
              grid_size: "20",
            })
          }
          disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white font-bold text-xl px-12 py-4 rounded-xl transition-all transform hover:scale-105"
        >
          {loading ? "Starting..." : "START SIMULATION"}
        </button>
      </div>
    </div>
  );
}
