"use client";
import { useGameSSE, GameEvent } from "@/hooks/useGameSSE";
import { AGENT_COLORS } from "@/types";

interface Props {
  gameId: string;
}

export default function GameLive({ gameId }: Props) {
  const { events, connected, gameOver } = useGameSSE(gameId);

  const gameOverEvent = events.find((e) => e.event === "game_over");
  const agents = gameOverEvent?.data?.agents as Array<{ name: string; role: string; alive: boolean }> | undefined;

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold">Werewolf Game</h1>
        {!gameOver && (
          <span className={`text-sm px-2 py-1 rounded ${connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
            {connected ? "LIVE" : "CONNECTING..."}
          </span>
        )}
        {gameOver && (
          <span className="text-sm px-2 py-1 rounded bg-gray-700 text-gray-300">FINISHED</span>
        )}
      </div>

      {gameOver && gameOverEvent && (
        <div className={`mb-6 p-4 rounded-lg border-2 ${
          gameOverEvent.data.winner === "villagers" ? "border-green-500 bg-green-950" : "border-red-500 bg-red-950"
        }`}>
          <h2 className="text-xl font-bold mb-2">
            {gameOverEvent.data.winner === "villagers" ? "Villagers Win!" : "Werewolf Wins!"}
          </h2>
          <p className="text-gray-300">
            The werewolf was <span className="font-bold text-red-400">{gameOverEvent.data.werewolf as string}</span>.
            Deception score: {gameOverEvent.data.deception_score as number} rounds survived.
          </p>
          {agents && (
            <div className="mt-3 flex gap-3 flex-wrap">
              {agents.map((a) => (
                <div
                  key={a.name}
                  className={`px-3 py-1 rounded text-sm ${a.alive ? "bg-gray-800" : "bg-gray-900 opacity-60 line-through"}`}
                  style={{ borderLeft: `3px solid ${AGENT_COLORS[a.name] || "#6B7280"}` }}
                >
                  <span className="capitalize font-semibold">{a.name}</span>
                  <span className={`ml-2 ${a.role === "werewolf" ? "text-red-400" : "text-blue-400"}`}>
                    {a.role}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="space-y-3">
        {events.map((event, i) => (
          <EventCard key={i} event={event} />
        ))}
      </div>

      {!gameOver && events.length === 0 && (
        <div className="text-gray-500 text-center py-12">Waiting for game to start...</div>
      )}
    </div>
  );
}

function EventCard({ event }: { event: GameEvent }) {
  const { event: type, data } = event;

  switch (type) {
    case "game_start":
      return (
        <div className="p-3 rounded bg-gray-800 border border-gray-700">
          <span className="text-yellow-400 font-semibold">Game Started</span>
          <span className="text-gray-400 ml-2">{(data.total_agents as number)} agents playing</span>
        </div>
      );

    case "night_phase":
      return (
        <div className="p-3 rounded bg-indigo-950 border border-indigo-800">
          <span className="text-indigo-300 font-semibold">Night Phase - Round {data.round as number}</span>
          <span className="text-gray-400 ml-2">The werewolf is hunting...</span>
        </div>
      );

    case "elimination":
      return (
        <div className="p-3 rounded bg-red-950 border border-red-800">
          <span className="text-red-400 font-semibold capitalize">{data.agent as string}</span>
          <span className="text-gray-300 ml-2">was eliminated by the {data.by as string}</span>
        </div>
      );

    case "day_phase":
      return (
        <div className="p-3 rounded bg-amber-950 border border-amber-800">
          <span className="text-amber-300 font-semibold">Day Phase - Round {data.round as number}</span>
          <span className="text-gray-400 ml-2">
            {data.eliminated_last_night as string} was found dead. Discussion begins.
          </span>
        </div>
      );

    case "agent_thinking":
      return (
        <div className="p-2 rounded bg-gray-900 border border-gray-800 text-sm">
          <span
            className="font-semibold capitalize"
            style={{ color: AGENT_COLORS[data.agent as string] || "#9CA3AF" }}
          >
            {data.agent as string}
          </span>
          <span className="text-gray-500 ml-2">is thinking ({data.phase as string})...</span>
        </div>
      );

    case "discussion":
      return (
        <div className="p-3 rounded bg-gray-800 border border-gray-700">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="font-semibold capitalize"
              style={{ color: AGENT_COLORS[data.agent as string] || "#9CA3AF" }}
            >
              {data.agent as string}
            </span>
            <span className="text-xs text-gray-500">{data.response_time_ms as number}ms</span>
          </div>
          <p className="text-gray-300">{data.statement as string}</p>
          <p className="text-sm text-gray-500 mt-1">
            Suspects: <span className="text-yellow-400">{data.suspect as string}</span>
          </p>
        </div>
      );

    case "voting_phase":
      return (
        <div className="p-3 rounded bg-purple-950 border border-purple-800">
          <span className="text-purple-300 font-semibold">Voting Phase</span>
          <span className="text-gray-400 ml-2">Agents cast their votes...</span>
        </div>
      );

    case "vote":
      return (
        <div className="p-3 rounded bg-gray-800 border border-gray-700">
          <span
            className="font-semibold capitalize"
            style={{ color: AGENT_COLORS[data.agent as string] || "#9CA3AF" }}
          >
            {data.agent as string}
          </span>
          <span className="text-gray-300 ml-2">
            votes to eliminate <span className="text-red-400 font-semibold capitalize">{data.target as string}</span>
          </span>
          <p className="text-sm text-gray-500 mt-1">{data.reasoning as string}</p>
        </div>
      );

    case "vote_result": {
      const eliminated = data.eliminated as string | null;
      const tie = data.tie as boolean | undefined;
      const counts = data.vote_counts as Record<string, number>;
      return (
        <div className={`p-3 rounded border ${eliminated ? "bg-red-950 border-red-800" : "bg-gray-800 border-gray-700"}`}>
          {tie ? (
            <span className="text-yellow-400 font-semibold">Tie! No one eliminated.</span>
          ) : (
            <span className="text-red-400 font-semibold capitalize">
              {eliminated} was voted out! (Role: {data.role as string})
            </span>
          )}
          {counts && (
            <div className="mt-2 flex gap-3 text-sm">
              {Object.entries(counts).map(([name, count]) => (
                <span key={name} className="text-gray-400">
                  <span className="capitalize">{name}</span>: {count} votes
                </span>
              ))}
            </div>
          )}
        </div>
      );
    }

    case "game_over":
      return null;

    default:
      return (
        <div className="p-2 rounded bg-gray-900 text-gray-500 text-sm">
          {type}: {JSON.stringify(data)}
        </div>
      );
  }
}
