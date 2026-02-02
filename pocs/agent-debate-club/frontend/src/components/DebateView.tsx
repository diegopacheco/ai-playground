import { useState, useRef } from 'react';
import { useDebateSSE } from '../hooks/useDebateSSE';
import { ChatWidget } from './ChatWidget';
import { Timer } from './Timer';
import { getAgentInfo } from '../types';

interface DebateViewProps {
  debateId: string;
  topic: string;
  duration: number;
  agentA: string;
  agentB: string;
  judge: string;
  onBack: () => void;
}

export function DebateView({ debateId, topic, duration, agentA, agentB, judge, onBack }: DebateViewProps) {
  const { messages, status, thinkingAgent, result, error } = useDebateSSE(debateId);
  const agentAInfo = getAgentInfo(agentA);
  const agentBInfo = getAgentInfo(agentB);
  const judgeInfo = getAgentInfo(judge);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const speechRef = useRef<SpeechSynthesisUtterance | null>(null);

  const isRunning = status === 'thinking' || (status === 'idle' && !result);

  const getWinnerName = (winner: string) => {
    if (winner === 'A') return agentAInfo.name;
    if (winner === 'B') return agentBInfo.name;
    return winner;
  };

  const getVoices = () => {
    const voices = window.speechSynthesis.getVoices();
    const english = voices.filter(v => v.lang.startsWith('en'));
    if (english.length >= 3) {
      return [english[0], english[1], english[2]];
    }
    if (voices.length >= 3) {
      return [voices[0], voices[1], voices[2]];
    }
    return [voices[0] || null, voices[1] || voices[0] || null, voices[2] || voices[0] || null];
  };

  const speakDebate = () => {
    if (isSpeaking) return;
    setIsSpeaking(true);
    const voices = getVoices();
    const textsToSpeak: { text: string; voiceIndex: number }[] = [];
    messages.forEach((msg) => {
      const agentName = msg.agent === 'A' ? agentAInfo.name : agentBInfo.name;
      textsToSpeak.push({ text: `${agentName} says: ${msg.content}`, voiceIndex: msg.agent === 'A' ? 0 : 1 });
    });
    if (result) {
      textsToSpeak.push({ text: `The judge ${judgeInfo.name} declares the winner is ${getWinnerName(result.winner)}. ${result.reason}`, voiceIndex: 2 });
    }
    let index = 0;
    const speakNext = () => {
      if (index >= textsToSpeak.length) {
        setIsSpeaking(false);
        return;
      }
      const item = textsToSpeak[index];
      const utterance = new SpeechSynthesisUtterance(item.text);
      if (voices[item.voiceIndex]) {
        utterance.voice = voices[item.voiceIndex];
      }
      utterance.rate = 1.0;
      utterance.onend = () => {
        index++;
        speakNext();
      };
      utterance.onerror = () => {
        setIsSpeaking(false);
      };
      speechRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    };
    speakNext();
  };

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <header className="bg-gray-800 p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={onBack}
            className="text-gray-400 hover:text-white transition"
          >
            Back
          </button>
          <h1 className="text-xl font-bold text-white flex-1 text-center px-4 truncate">
            {topic}
          </h1>
          <div className="text-white">
            <Timer durationSeconds={duration} isRunning={isRunning} />
          </div>
        </div>
        <div className="max-w-4xl mx-auto mt-2 flex items-center justify-center gap-4 text-sm">
          <span style={{ color: agentAInfo.color }}>{agentAInfo.name} ({agentAInfo.model})</span>
          <span className="text-gray-500">vs</span>
          <span style={{ color: agentBInfo.color }}>{agentBInfo.name} ({agentBInfo.model})</span>
          <span className="text-gray-500">|</span>
          <span className="text-gray-400">Judge: <span style={{ color: judgeInfo.color }}>{judgeInfo.name} ({judgeInfo.model})</span></span>
          <span className="text-gray-500">|</span>
          {!isSpeaking ? (
            <button
              onClick={speakDebate}
              disabled={messages.length === 0}
              className="px-3 py-1 bg-purple-600 text-white rounded text-xs font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Read Debate
            </button>
          ) : (
            <button
              onClick={stopSpeaking}
              className="px-3 py-1 bg-red-600 text-white rounded text-xs font-medium hover:bg-red-700 transition"
            >
              Stop
            </button>
          )}
        </div>
      </header>

      <div className="flex-1 max-w-4xl mx-auto w-full flex flex-col">
        <ChatWidget
          messages={messages}
          thinkingAgent={thinkingAgent}
          agentAInfo={agentAInfo}
          agentBInfo={agentBInfo}
        />

        {error && (
          <div className="bg-red-600 text-white p-4 text-center">{error}</div>
        )}

        {result && (
          <div className="bg-gray-800 p-6 border-t border-gray-700">
            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-2">
                Winner: {getWinnerName(result.winner)}
              </div>
              <p className="text-gray-300">{result.reason}</p>
              <p className="text-gray-500 text-sm mt-2">
                Duration: {Math.round(result.duration_ms / 1000)}s | Judged by {judgeInfo.name}
              </p>
              <button
                onClick={onBack}
                className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
              >
                New Debate
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
