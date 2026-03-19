import { useState, useEffect, useRef } from "react";
import SetupPanel from "~/components/SetupPanel";
import GameGrid from "~/components/GameGrid";
import StatsPanel from "~/components/StatsPanel";
import HistoryTable from "~/components/HistoryTable";
import { useGameSSE } from "~/hooks/useGameSSE";
import { useSound } from "~/hooks/useSound";
import { createGame, stopGame } from "~/api/game";

type Tab = "setup" | "simulation" | "history";

export default function Index() {
  const [tab, setTab] = useState<Tab>("setup");
  const [gameId, setGameId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [terminatorAgent, setTerminatorAgent] = useState("");
  const [terminatorModel, setTerminatorModel] = useState("");
  const [mosquitoAgent, setMosquitoAgent] = useState("");
  const [mosquitoModel, setMosquitoModel] = useState("");

  const state = useGameSSE(gameId);
  const { play, muted, toggleMute } = useSound();
  const prevCycle = useRef(0);

  useEffect(() => {
    if (state.cycle === prevCycle.current) return;
    prevCycle.current = state.cycle;

    if (state.recentKills.length > 0) play("kill");
    if (state.recentDates.length > 0) play("date");
    if (state.recentHatches.length > 0) play("hatch");
  }, [state.cycle, state.recentKills, state.recentDates, state.recentHatches, play]);

  useEffect(() => {
    if (state.status === "finished") {
      if (state.winner === "terminator") play("victory");
      else if (state.winner === "mosquitos") play("swarm");
    }
  }, [state.status, state.winner, play]);

  const handleStart = async (config: {
    terminator_agent: string;
    terminator_model: string;
    mosquito_agent: string;
    mosquito_model: string;
    grid_size: string;
  }) => {
    setLoading(true);
    setTerminatorAgent(config.terminator_agent);
    setTerminatorModel(config.terminator_model);
    setMosquitoAgent(config.mosquito_agent);
    setMosquitoModel(config.mosquito_model);
    try {
      const result = await createGame(config);
      setGameId(result.id);
      setTab("simulation");
      play("start");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (gameId) await stopGame(gameId);
  };

  return (
    <div className="min-h-screen">
      <nav className="border-b border-gray-800 bg-[#071a10]">
        <div className="max-w-7xl mx-auto flex">
          {(["setup", "simulation", "history"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-6 py-4 text-sm font-bold uppercase tracking-wider transition-colors ${
                tab === t
                  ? "text-white border-b-2 border-white"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {t === "setup" ? "🤖 Setup" : t === "simulation" ? "🎮 Simulation" : "📊 History"}
            </button>
          ))}
        </div>
      </nav>

      {tab === "setup" && <SetupPanel onStart={handleStart} loading={loading} />}

      {tab === "simulation" && (
        <div className="max-w-7xl mx-auto p-6">
          {!gameId ? (
            <div className="text-center text-gray-500 py-20">
              Go to Setup tab to start a simulation
            </div>
          ) : (
            <div className="flex gap-6">
              <div className="flex-shrink-0">
                <GameGrid state={state} />
              </div>
              <div className="flex-1 min-w-[300px]">
                <StatsPanel
                  state={state}
                  terminatorAgent={terminatorAgent}
                  terminatorModel={terminatorModel}
                  mosquitoAgent={mosquitoAgent}
                  mosquitoModel={mosquitoModel}
                  muted={muted}
                  onToggleMute={toggleMute}
                  onStop={handleStop}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {tab === "history" && <HistoryTable />}
    </div>
  );
}
