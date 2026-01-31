import { useEffect, useRef } from 'react';
import type { Message, AgentInfo } from '../types';

interface ChatWidgetProps {
  messages: Message[];
  thinkingAgent: string | null;
  agentAInfo: AgentInfo;
  agentBInfo: AgentInfo;
}

export function ChatWidget({ messages, thinkingAgent, agentAInfo, agentBInfo }: ChatWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, thinkingAgent]);

  const getAgentDisplay = (agent: string) => {
    const info = agent === 'A' ? agentAInfo : agentBInfo;
    return { name: info.name, model: info.model, color: info.color };
  };

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50"
    >
      {messages.length === 0 && !thinkingAgent && (
        <div className="text-center text-gray-400 py-8">
          Waiting for debate to start...
        </div>
      )}

      {messages.map((msg) => {
        const agentDisplay = getAgentDisplay(msg.agent);
        return (
          <div
            key={msg.id}
            className={`p-4 rounded-lg ${
              msg.agent === 'A'
                ? 'bg-blue-100 ml-0 mr-8'
                : 'bg-green-100 ml-8 mr-0'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="font-bold" style={{ color: agentDisplay.color }}>
                {agentDisplay.name}
              </span>
              <span className="text-xs text-gray-500">({agentDisplay.model})</span>
              <span
                className={`text-xs px-2 py-1 rounded ${
                  msg.stance === 'ATTACK'
                    ? 'bg-red-200 text-red-700'
                    : 'bg-blue-200 text-blue-700'
                }`}
              >
                {msg.stance}
              </span>
            </div>
            <p className="text-gray-800">{msg.content}</p>
          </div>
        );
      })}

      {thinkingAgent && (
        <div
          className={`p-4 rounded-lg ${
            thinkingAgent === 'A'
              ? 'bg-blue-50 ml-0 mr-8'
              : 'bg-green-50 ml-8 mr-0'
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="font-bold" style={{ color: getAgentDisplay(thinkingAgent).color }}>
              {getAgentDisplay(thinkingAgent).name}
            </span>
            <span className="text-xs text-gray-500">({getAgentDisplay(thinkingAgent).model})</span>
            <span className="text-gray-400 animate-pulse">thinking...</span>
          </div>
        </div>
      )}
    </div>
  );
}
