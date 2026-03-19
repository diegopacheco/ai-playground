import { useState } from "react";
import { AVAILABLE_AGENTS } from "../types/index.ts";
import type { Agent } from "../types/index.ts";

interface AuctionSetupProps {
  onStart: (agents: Agent[]) => void;
  isLoading: boolean;
}

interface AgentConfig {
  name: string;
  model: string;
  budget: number;
  selected: boolean;
}

export function AuctionSetup({ onStart, isLoading }: AuctionSetupProps) {
  const [configs, setConfigs] = useState<AgentConfig[]>(
    AVAILABLE_AGENTS.map((a) => ({
      name: a.name,
      model: a.defaultModel,
      budget: 100,
      selected: false,
    }))
  );

  const selectedCount = configs.filter((c) => c.selected).length;

  function toggleAgent(name: string) {
    setConfigs((prev) =>
      prev.map((c) => {
        if (c.name !== name) return c;
        if (!c.selected && selectedCount >= 3) return c;
        return { ...c, selected: !c.selected };
      })
    );
  }

  function updateModel(name: string, model: string) {
    setConfigs((prev) =>
      prev.map((c) => (c.name === name ? { ...c, model } : c))
    );
  }

  function updateBudget(name: string, budget: number) {
    setConfigs((prev) =>
      prev.map((c) => (c.name === name ? { ...c, budget } : c))
    );
  }

  function handleStart() {
    const agents = configs
      .filter((c) => c.selected)
      .map((c) => ({ name: c.name, model: c.model, budget: c.budget }));
    onStart(agents);
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold text-amber-400 mb-2 text-center">
        Agent Auction House
      </h2>
      <p className="text-gray-400 text-center mb-8">
        Select exactly 3 agents to compete in the auction
      </p>

      <div className="grid gap-4">
        {AVAILABLE_AGENTS.map((agentOption) => {
          const config = configs.find((c) => c.name === agentOption.name)!;
          return (
            <div
              key={agentOption.name}
              className={`rounded-lg p-4 border-2 cursor-pointer transition-all ${
                config.selected
                  ? "bg-gray-800 border-opacity-100"
                  : "bg-gray-800/50 border-gray-700 border-opacity-50"
              }`}
              style={{
                borderColor: config.selected ? agentOption.color : undefined,
              }}
              onClick={() => toggleAgent(agentOption.name)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={config.selected}
                    onChange={() => toggleAgent(agentOption.name)}
                    className="w-5 h-5 accent-amber-500"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <span
                    className="font-bold text-xl capitalize"
                    style={{ color: agentOption.color }}
                  >
                    {agentOption.name}
                  </span>
                </div>
              </div>

              {config.selected && (
                <div
                  className="mt-4 flex gap-4 items-end"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex-1">
                    <label className="text-gray-400 text-sm block mb-1">
                      Model
                    </label>
                    <select
                      value={config.model}
                      onChange={(e) => updateModel(agentOption.name, e.target.value)}
                      className="w-full bg-gray-700 text-white rounded px-3 py-2 border border-gray-600"
                    >
                      {agentOption.models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="w-32">
                    <label className="text-gray-400 text-sm block mb-1">
                      Budget ($)
                    </label>
                    <input
                      type="number"
                      value={config.budget}
                      onChange={(e) =>
                        updateBudget(agentOption.name, parseInt(e.target.value) || 0)
                      }
                      min={10}
                      max={1000}
                      className="w-full bg-gray-700 text-white rounded px-3 py-2 border border-gray-600"
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 text-center">
        <button
          onClick={handleStart}
          disabled={selectedCount !== 3 || isLoading}
          className="bg-amber-600 hover:bg-amber-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-3 px-8 rounded-lg text-lg transition-colors"
        >
          {isLoading ? "Starting..." : `Start Auction (${selectedCount}/3 selected)`}
        </button>
      </div>
    </div>
  );
}
