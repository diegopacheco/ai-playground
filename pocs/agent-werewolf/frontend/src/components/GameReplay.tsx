"use client";
import { Game, getAgentColor } from "@/types";

interface Props {
  game: Game;
}

export default function GameReplay({ game }: Props) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold">Werewolf Game</h1>
        <span className="text-sm px-2 py-1 rounded bg-gray-700 text-gray-300">FINISHED</span>
      </div>

      <div className={`mb-6 p-4 rounded-lg border-2 ${
        game.winner === "villagers" ? "border-green-500 bg-green-950" : "border-red-500 bg-red-950"
      }`}>
        <h2 className="text-xl font-bold mb-2">
          {game.winner === "villagers" ? "Villagers Win!" : "Werewolf Wins!"}
        </h2>
        <p className="text-gray-300">
          The werewolf was <span className="font-bold text-red-400">{game.werewolf_agent}</span>.
          Deception score: {game.deception_score} rounds survived.
        </p>
        <div className="mt-3 flex gap-3 flex-wrap">
          {game.agents.map((a) => (
            <div
              key={a.id}
              className={`px-3 py-1 rounded text-sm ${a.alive ? "bg-gray-800" : "bg-gray-900 opacity-60 line-through"}`}
              style={{ borderLeft: `3px solid ${getAgentColor(a.agent_name)}` }}
            >
              <span className="capitalize font-semibold">{a.agent_name}</span>
              <span className={`ml-2 ${a.role === "werewolf" ? "text-red-400" : "text-blue-400"}`}>
                {a.role}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {game.rounds.map((round) => (
          <div key={round.id}>
            <div className={`p-3 rounded border ${
              round.phase === "night"
                ? "bg-indigo-950 border-indigo-800"
                : "bg-amber-950 border-amber-800"
            }`}>
              <span className={round.phase === "night" ? "text-indigo-300 font-semibold" : "text-amber-300 font-semibold"}>
                {round.phase === "night" ? "Night" : "Day"} Phase - Round {round.round_number}
              </span>
              {round.eliminated_agent && (
                <span className="text-gray-400 ml-2">
                  {round.eliminated_agent} was eliminated by {round.eliminated_by}
                </span>
              )}
            </div>

            {round.messages.map((msg) => (
              <MessageCard key={msg.id} msg={msg} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function MessageCard({ msg }: { msg: { agent_name: string; message_type: string; content: string; target: string | null; response_time_ms: number | null } }) {
  switch (msg.message_type) {
    case "night_kill":
      return (
        <div className="p-3 rounded bg-red-950 border border-red-800 mt-2">
          <span className="text-red-400 font-semibold capitalize">{msg.target}</span>
          <span className="text-gray-300 ml-2">was eliminated by the werewolf</span>
        </div>
      );

    case "discussion":
      return (
        <div className="p-3 rounded bg-gray-800 border border-gray-700 mt-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold capitalize" style={{ color: getAgentColor(msg.agent_name) }}>
              {msg.agent_name}
            </span>
            {msg.response_time_ms && (
              <span className="text-xs text-gray-500">{msg.response_time_ms}ms</span>
            )}
          </div>
          <p className="text-gray-300">{msg.content}</p>
          {msg.target && (
            <p className="text-sm text-gray-500 mt-1">
              Suspects: <span className="text-yellow-400">{msg.target}</span>
            </p>
          )}
        </div>
      );

    case "vote":
      return (
        <div className="p-3 rounded bg-gray-800 border border-gray-700 mt-2">
          <span className="font-semibold capitalize" style={{ color: getAgentColor(msg.agent_name) }}>
            {msg.agent_name}
          </span>
          <span className="text-gray-300 ml-2">
            votes to eliminate <span className="text-red-400 font-semibold capitalize">{msg.target}</span>
          </span>
          <p className="text-sm text-gray-500 mt-1">{msg.content}</p>
        </div>
      );

    default:
      return null;
  }
}
