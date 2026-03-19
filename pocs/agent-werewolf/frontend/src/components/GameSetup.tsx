"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AgentInfo, AgentSelection, getAgentColor } from "@/types";
import { getAgents, createGame } from "@/lib/api";

export default function GameSetup() {
  const router = useRouter();
  const [availableAgents, setAvailableAgents] = useState<AgentInfo[]>([]);
  const [slots, setSlots] = useState<AgentSelection[]>([
    { name: "", model: "" },
    { name: "", model: "" },
    { name: "", model: "" },
    { name: "", model: "" },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getAgents().then(setAvailableAgents).catch(() => setError("Backend not reachable at localhost:3000. Make sure it is running."));
  }, []);

  function updateSlot(index: number, name: string) {
    const agent = availableAgents.find((a) => a.name === name);
    const updated = [...slots];
    updated[index] = { name, model: agent?.default_model || "" };
    setSlots(updated);
  }

  function updateSlotModel(index: number, model: string) {
    const updated = [...slots];
    updated[index] = { ...updated[index], model };
    setSlots(updated);
  }

  function addSlot() {
    if (slots.length < 6) {
      setSlots([...slots, { name: "", model: "" }]);
    }
  }

  function removeSlot(index: number) {
    if (slots.length > 4) {
      setSlots(slots.filter((_, i) => i !== index));
    }
  }

  const allFilled = slots.every((s) => s.name !== "");

  async function startGame() {
    if (!allFilled) return;
    setLoading(true);
    setError("");
    try {
      const result = await createGame(slots);
      router.push(`/game/${result.id}`);
    } catch {
      setError("Failed to create game. Make sure the backend is running on localhost:3000.");
      setLoading(false);
    }
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">New Werewolf Game</h1>
      <p className="text-gray-400 mb-6">
        Select 4-6 agents. You can pick the same agent multiple times. One will be randomly assigned as the werewolf.
      </p>

      <div className="space-y-3 mb-6">
        {slots.map((slot, i) => {
          const agent = availableAgents.find((a) => a.name === slot.name);
          const color = getAgentColor(slot.name);
          return (
            <div key={i} className="flex items-center gap-3">
              <span className="text-gray-500 w-8 text-right">#{i + 1}</span>
              <select
                value={slot.name}
                onChange={(e) => updateSlot(i, e.target.value)}
                className="bg-gray-800 border border-gray-700 text-white rounded px-3 py-2 w-48"
                style={slot.name ? { borderColor: color } : {}}
              >
                <option value="">Select agent...</option>
                {availableAgents.map((a) => (
                  <option key={a.name} value={a.name}>{a.name}</option>
                ))}
              </select>
              {agent && (
                <select
                  value={slot.model}
                  onChange={(e) => updateSlotModel(i, e.target.value)}
                  className="bg-gray-800 border border-gray-700 text-white rounded px-3 py-2 w-56"
                >
                  {agent.models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              )}
              {slots.length > 4 && (
                <button
                  onClick={() => removeSlot(i)}
                  className="text-gray-500 hover:text-red-400 px-2"
                >
                  X
                </button>
              )}
            </div>
          );
        })}
      </div>

      {slots.length < 6 && (
        <button
          onClick={addSlot}
          className="mb-6 px-4 py-2 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 rounded transition-colors"
        >
          + Add Agent Slot
        </button>
      )}

      {error && (
        <div className="mb-4 p-3 rounded bg-red-950 border border-red-800 text-red-400">{error}</div>
      )}

      <div className="flex items-center gap-4">
        <button
          onClick={startGame}
          disabled={!allFilled || loading}
          className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-semibold transition-colors"
        >
          {loading ? "Starting..." : `Start Game (${slots.length} agents)`}
        </button>
        {!allFilled && (
          <span className="text-gray-500">Fill all agent slots</span>
        )}
      </div>
    </div>
  );
}
