import type { GameState } from "~/hooks/useGameSSE";

interface Props {
  state: GameState;
  terminatorAgent: string;
  terminatorModel: string;
  mosquitoAgent: string;
  mosquitoModel: string;
  muted: boolean;
  onToggleMute: () => void;
  onStop: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function StatsPanel({
  state,
  terminatorAgent,
  terminatorModel,
  mosquitoAgent,
  mosquitoModel,
  muted,
  onToggleMute,
  onStop,
}: Props) {
  return (
    <div className="bg-[#0d2818] border border-gray-800 rounded-xl p-5 flex flex-col gap-4 h-full">
      <div className="text-center">
        <div className="text-5xl font-mono font-bold text-white tracking-wider">
          {formatTime(state.cycle)}
        </div>
        <div className="text-sm text-gray-500 mt-1">
          Cycle {state.cycle} / 200
        </div>
        {state.status === "finished" && (
          <div
            className={`mt-2 text-xl font-bold ${
              state.winner === "terminator"
                ? "text-red-500"
                : state.winner === "mosquitos"
                  ? "text-green-500"
                  : "text-yellow-500"
            }`}
          >
            {state.winner === "terminator"
              ? "TERMINATED!"
              : state.winner === "mosquitos"
                ? "MOSQUITOS WIN!"
                : state.winner === "draw"
                  ? "DRAW!"
                  : state.winner?.toUpperCase()}
          </div>
        )}
      </div>

      <div className="border-t border-gray-800 pt-3">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xl">🤖</span>
          <div>
            <div className="text-red-400 font-bold text-sm">{terminatorAgent}</div>
            <div className="text-gray-500 text-xs">{terminatorModel}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xl">🦟</span>
          <div>
            <div className="text-green-400 font-bold text-sm">{mosquitoAgent}</div>
            <div className="text-gray-500 text-xs">{mosquitoModel}</div>
          </div>
        </div>
      </div>

      <div className="border-t border-gray-800 pt-3 space-y-3">
        <StatRow label="Terminator" value="1" color="text-red-400" icon="🤖" />
        <StatRow
          label="Mosquitos Alive"
          value={String(state.aliveCount)}
          color="text-green-400"
          icon="🦟"
        />
        <StatRow
          label="Eggs on Grid"
          value={String(state.eggCount)}
          color="text-yellow-400"
          icon="🥚"
        />
        <div className="border-t border-gray-800 pt-2" />
        <StatRow
          label="Total Kills"
          value={String(state.totalKills)}
          color="text-red-300"
          icon="💀"
        />
        <StatRow
          label="Total Hatched"
          value={String(state.totalHatched)}
          color="text-yellow-300"
          icon="🐣"
        />
        <StatRow
          label="Total Dates"
          value={String(state.totalDates)}
          color="text-pink-300"
          icon="💕"
        />
      </div>

      <div className="border-t border-gray-800 pt-3 flex-1 min-h-0">
        <h3 className="text-sm font-bold text-gray-400 mb-2">EVENT LOG</h3>
        <div className="overflow-y-auto max-h-48 space-y-1 text-xs font-mono">
          {state.events.map((event, i) => (
            <div
              key={i}
              className={`${
                event.includes("TERMINATED")
                  ? "text-red-400"
                  : event.includes("dating")
                    ? "text-pink-400"
                    : event.includes("hatched")
                      ? "text-yellow-400"
                      : event.includes("died")
                        ? "text-gray-600"
                        : event.includes("GAME OVER")
                          ? "text-white font-bold"
                          : "text-gray-400"
              }`}
            >
              {event}
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-gray-800 pt-3 flex gap-2">
        <button
          onClick={onToggleMute}
          className="flex-1 bg-gray-800 hover:bg-gray-700 text-white py-2 px-4 rounded-lg text-sm transition-colors"
        >
          {muted ? "🔇 Unmute" : "🔊 Mute"}
        </button>
        {state.status === "running" && (
          <button
            onClick={onStop}
            className="flex-1 bg-red-900 hover:bg-red-800 text-white py-2 px-4 rounded-lg text-sm transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}

function StatRow({
  label,
  value,
  color,
  icon,
}: {
  label: string;
  value: string;
  color: string;
  icon: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <span className={`text-2xl font-mono font-bold ${color}`}>{value}</span>
    </div>
  );
}
