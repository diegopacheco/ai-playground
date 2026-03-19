"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AgentInfo, AgentSelection, AGENT_COLORS } from "@/types";
import { getAgents, createGame } from "@/lib/api";

export default function GameSetup() {
  const router = useRouter();
  const [availableAgents, setAvailableAgents] = useState<AgentInfo[]>([]);
  const [selected, setSelected] = useState<AgentSelection[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getAgents().then(setAvailableAgents).catch(() => {});
  }, []);

  function toggleAgent(agent: AgentInfo) {
    const exists = selected.find((s) => s.name === agent.name);
    if (exists) {
      setSelected(selected.filter((s) => s.name !== agent.name));
    } else if (selected.length < 6) {
      setSelected([...selected, { name: agent.name, model: agent.default_model }]);
    }
  }

  function updateModel(name: string, model: string) {
    setSelected(selected.map((s) => (s.name === name ? { ...s, model } : s)));
  }

  async function startGame() {
    if (selected.length < 4) return;
    setLoading(true);
    const result = await createGame(selected);
    router.push(`/game/${result.id}`);
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">New Werewolf Game</h1>
      <p className="text-gray-400 mb-6">
        Select 4-6 agents. One will be randomly assigned as the werewolf.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {availableAgents.map((agent) => {
          const isSelected = selected.some((s) => s.name === agent.name);
          const color = AGENT_COLORS[agent.name] || "#6B7280";
          return (
            <div
              key={agent.name}
              onClick={() => toggleAgent(agent)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                isSelected
                  ? "border-current bg-gray-800"
                  : "border-gray-700 bg-gray-900 hover:border-gray-500"
              }`}
              style={isSelected ? { borderColor: color } : {}}
            >
              <div className="text-lg font-semibold capitalize" style={{ color }}>
                {agent.name}
              </div>
              {isSelected && (
                <select
                  value={selected.find((s) => s.name === agent.name)?.model}
                  onChange={(e) => {
                    e.stopPropagation();
                    updateModel(agent.name, e.target.value);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className="mt-2 w-full bg-gray-700 text-white text-sm rounded px-2 py-1"
                >
                  {agent.models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={startGame}
          disabled={selected.length < 4 || loading}
          className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-semibold transition-colors"
        >
          {loading ? "Starting..." : `Start Game (${selected.length}/4-6 agents)`}
        </button>
        {selected.length < 4 && (
          <span className="text-gray-500">Select at least 4 agents</span>
        )}
      </div>
    </div>
  );
}
